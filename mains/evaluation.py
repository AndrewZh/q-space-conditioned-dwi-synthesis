import sys
sys.path.append('..')
sys.path.append('.')
import torch
from torch.autograd import Variable
from mains.trainer import dwi_Trainer
from utils.utilization import get_config
import numpy as np
import os
import os.path as osp
import nibabel as nib
import dipy.io as dio
from dipy.reconst.shm import normalize_data
import h5py
import skimage.io as skio

def synthesize_slice_per_bvec_bval(norm_b0, t2, t1, s, net, direction, bvalue, device):
    with torch.no_grad():
        x, y, _ = norm_b0.shape
        out2d = np.zeros([x, y])
        in_c = 0
        in_slice = None
        if norm_b0 is not None:
            # b0_slice = norm_b0
            b0_slice = norm_b0[..., s].transpose()
            in_c += 1
            in_slice = np.expand_dims(b0_slice, -1).astype(np.float32)
        if t2 is not None:
            in_c += 1
            # t2_slice = t2
            t2_slice = t2[..., s].transpose()
            t2_slice = t2_slice / (1e-6 + t2_slice.max())
            t2_slice = np.clip(t2_slice, 0., 1.)
            t2_slice[b0_slice==0] = 0
            t2_slice = np.expand_dims(t2_slice, -1).astype(np.float32)
            if in_slice is None:
                in_slice = t2_slice
            else:
                in_slice = np.concatenate((in_slice, t2_slice), axis=-1)
        if t1 is not None:
            in_c += 1
            # t1_slice = t1
            t1_slice = t1[..., s].transpose()
            t1_slice = t1_slice / (1e-6 + t1_slice.max())
            t1_slice = np.clip(t1_slice, 0., 1.)
            t1_slice[b0_slice==0] = 0
            t1_slice = np.expand_dims(t1_slice, -1).astype(np.float32)
            if in_slice is None:
                in_slice = t1_slice
            else:
                in_slice = np.concatenate((in_slice, t1_slice), axis=-1)

        raw_x, raw_y = in_slice.shape[0], in_slice.shape[1]
        xs = np.zeros([180, 180, in_c])
        xs[:raw_x, :raw_y, :] = in_slice

        xs = xs.transpose(2, 0, 1) #????? is it correct?

        input_ = Variable(torch.from_numpy(xs.copy())).unsqueeze(0).cuda().float().to(device)
        cond = Variable(torch.from_numpy(direction.copy())).unsqueeze(0).cuda().float().to(device)
        bv = Variable(torch.from_numpy(np.array([bvalue]).copy())).unsqueeze(0).cuda().float().to(device)
        cond = torch.cat((cond, bv), dim=-1).to(device)
        
        output = net(input_, cond)
        
        mask = (input_[:, 0, :, :] > 0.).float().expand_as(output)
        output = output * mask
        out_np = output[0, 0, :, :].cpu().numpy()
        out_np = out_np[:raw_x, :raw_y]
        out_np = out_np.transpose()
        out_np[out_np < 0] = 0
        # out_np = out_np * (out_np - out_np.min()) / (out_np.max() - out_np.min())
        out_np = np.clip(out_np, 0, 1)
        out2d = out_np

        del input_
        del cond
        del bv
        
        return out2d

def ref_loading(folder, pad_size=180, key='test', dir_group=2):
    idx = 596
    hf = h5py.File(folder, 'r')
    if dir_group > 0:
        idx = np.arange(idx, idx+dir_group).tolist()
    slice_b0 = hf['{}_b0'.format(key)][idx].transpose(0,2,1)
    slice_dwis = hf['{}_dwi'.format(key)][idx].transpose(0,2,1)
    t1 =  hf['{}_t1'.format(key)][idx].transpose(0,2,1)
    t2 =  hf['{}_t2'.format(key)][idx].transpose(0,2,1)
    t2[t2 < 0.] = 0.
    t1[t1 < 0.] = 0.
    t1 /= np.max(t1, (1,2))[:,None, None]
    t2 /= np.max(t2, (1,2))[:,None, None]
    t1[slice_b0 == 0.] = 0.
    t2[slice_b0 == 0.] = 0.
    slice_b0 = np.clip(slice_b0, 0, 1)
    cond = hf['{}_bval_vec'.format(key)][idx]
    print(cond)
    bvals_ = cond[:, 3, None, None]

    cond[:,3]  /= 3010
    
    assert dir_group == slice_dwis.shape[0]

    b0_list = [np.expand_dims(slice_b0[i], -1).astype(np.float32) for i in range(dir_group)]
    t2_list = [np.expand_dims(t2[i], -1).astype(np.float32) for i in range(dir_group)]
    t1_list = [np.expand_dims(t1[i], -1).astype(np.float32) for i in range(dir_group)]
    dwi_list = [np.expand_dims(slice_dwis[i], -1).astype(np.float32) for i in range(dir_group)]
    cond_list = [cond[i].astype(np.float32) for i in range(dir_group)]

    return_dict = dict()
    pad_size = pad_size
    w, h, _ = b0_list[0].shape
    for i in range(dir_group):
        pad_xs = np.zeros((pad_size, pad_size, 1))
        pad_xs[:w, :h] = b0_list[i]
        b0_list[i] = pad_xs

        pad_ys = np.zeros_like(pad_xs)
        pad_ys[:w,:h] = dwi_list[i]
        dwi_list[i] = pad_ys

        pad_t2 = np.zeros_like(pad_xs)
        pad_t2[:w,:h] = t2_list[i]
        t2_list[i] = pad_t2

        pad_t1 = np.zeros_like(pad_xs)
        pad_t1[:w,:h] = t1_list[i]
        t1_list[i] = pad_t1

    for i in range(dir_group):
        return_dict['b0_%d'%(i+1)] = b0_list[i].transpose(2, 0, 1)
        return_dict['t2_%d'%(i+1)] = t2_list[i].transpose(2, 0, 1)
        return_dict['t1_%d'%(i+1)] = t1_list[i].transpose(2, 0, 1)
        return_dict['dwi_%d'%(i+1)] = dwi_list[i].transpose(2, 0, 1)
        return_dict['cond_%d'%(i+1)] = cond_list[i]

    return return_dict

if __name__ == '__main__':
    config = get_config('/data/andrey/q-space-conditioned-dwi-synthesis/configs/smri2dwi.yaml')
    checkpoint = '/data/andrey/q-space-conditioned-dwi-synthesis/logs/smri2dwi_noSkull_NoStructNorm/gen_latest.pt'

    trainer = dwi_Trainer(config)
    net = trainer.gen_a

    state_dict = torch.load(checkpoint, map_location=trainer.device)
    device=trainer.device
    net.load_state_dict(state_dict['a'])
    _ = net.eval()

    data_dir = '/data/s2ms/test' 
    out_dir = "qsampling"
    subj_source_dir = '/data/s2ms/results/b0_n_5xb1000_to_b2000'
    subjects = sorted([s for s in os.listdir(subj_source_dir) if osp.isdir(osp.join(subj_source_dir, s))])

    # val_file = '/data/s2ms/test/test_1_nonPad_NoStructNorm_4Qsampl.hdf5'
    
    # return_dict = ref_loading(val_file)
    # dwi_h5 = np.squeeze(return_dict['dwi_1'])
    # skio.imsave('dwi_h5.png', dwi_h5.transpose())

    for subj_id in subjects:
        print(subj_id)
        dwi_img = nib.load(f'/data/s2ms/test/{subj_id}/data.nii.gz')
        dwi = dwi_img.get_fdata()
        mask = nib.load(f'/data/s2ms/test/{subj_id}/nodif_brain_mask.nii.gz').get_fdata()
        dwi[mask==0,:] = 0
        bvals, bvecs = dio.read_bvals_bvecs(f'/data/s2ms/test/{subj_id}/bvals', f'/data/s2ms/test/{subj_id}/bvecs')
        t1 = nib.load(f'/data/s2ms/test/{subj_id}/{subj_id}_t1_125.nii.gz').get_fdata()
        t1[mask==0] = 0
        # t1 /= t1.max()
        t2 = nib.load(f'/data/s2ms/test/{subj_id}/{subj_id}_t2_125.nii.gz').get_fdata()
        t2[mask==0] = 0
        # t2 /= t2.max() 

        rbval = (np.round(bvals/1000)*1000).astype(np.int32)
        b0_mask = rbval == 0
        b1000_mask = rbval == 1000
        b2000_mask = rbval == 2000

        num_b0 = np.sum(b0_mask)
        num_b1000 = np.sum(b1000_mask)
        num_b2000 = np.sum(b2000_mask)
        num_vols = num_b0 + num_b1000 + num_b2000

        norm_dwi = normalize_data(dwi, b0_mask)
        norm_dwi[dwi==0] = 0
        norm_b1000 = norm_dwi[..., b1000_mask]
        norm_b1000[dwi[..., b1000_mask]<1e-5] = 0
        norm_b1000[norm_b1000 < 0] = 0
        norm_b1000 = np.clip(norm_b1000, 0, 1)

        b0_vols = dwi[..., b0_mask]
        b0_avrg_vol = np.mean(b0_vols, axis=-1)
        b0_max = b0_avrg_vol.max()
        b0_avrg_vol = b0_avrg_vol / b0_max
        b0_avrg_vol[b0_avrg_vol < 0] = 0

        bvals_2000 = bvals[b2000_mask] / 3010 #np.max(bvals)
        bvecs_2000 = bvecs[b2000_mask,:] 

        # dwi_2000 = norm_dwi[..., b2000_mask]
        # dwi_2000[dwi[..., b2000_mask]==0] = 0
        # dwi_2000[dwi_2000 < 0] = 0
        # skio.imsave('dwi_gt.png', np.clip(np.squeeze(norm_dwi[...,42,6]), 0, 1))
        
        gen_dwi = np.zeros(dwi.shape[:-1]+(num_vols,))

        if not osp.isdir(f'/data/s2ms/results/{out_dir}/{subj_id}/'):
            os.makedirs(f'/data/s2ms/results/{out_dir}/{subj_id}/')

        gen_bval = np.concatenate((bvals[b0_mask], bvals[b1000_mask], bvals[b2000_mask])).astype(int)
        gen_bvec = np.concatenate((bvecs[b0_mask,:], bvecs[b1000_mask,:], bvecs[b2000_mask,:]))

        with open(f'/data/s2ms/results/{out_dir}/{subj_id}/bvals','w') as gen_bval_file:
            gen_bval_file.write(' '.join(map(str, gen_bval)))
            gen_bval_file.write('\n')
        with open(f'/data/s2ms/results/{out_dir}/{subj_id}/bvecs','w') as gen_bvec_file:
            gen_bvec_file.write(' '.join(map(str, gen_bvec[:,0])))
            gen_bvec_file.write('\n')
            gen_bvec_file.write(' '.join(map(str, gen_bvec[:,1])))
            gen_bvec_file.write('\n')
            gen_bvec_file.write(' '.join(map(str, gen_bvec[:,2])))
            gen_bvec_file.write('\n')
        
        gen_dwi[..., :num_b0] = b0_vols
        gen_dwi[..., num_b0: num_b0+num_b1000] = norm_b1000
        # b1000_sample = dwi[...,b1000_mask][...,0]
        
        for volIdx in range(num_b2000):
            # volIdx=1
            for sliceIdx in range(dwi.shape[2]):
                # sliceIdx=42
                print('Volume', volIdx,'Slice', sliceIdx, end='\r')
                if np.sum(mask[...,sliceIdx]) > 0:
                    gen_slice = synthesize_slice_per_bvec_bval(b0_avrg_vol, t2, t1,
                                            sliceIdx, net, bvecs_2000[volIdx,:], bvals_2000[volIdx], device)
                    
                    # skio.imsave('dwi_gen.png', np.clip(gen_slice, 0, 1))
                    gen_dwi[:,:,sliceIdx, num_b0+num_b1000+volIdx] = np.clip(gen_slice, 0, 1)
                    # gen_slice = synthesize_slice_per_bvec_bval(np.squeeze(return_dict['b0_1']), np.squeeze(return_dict['t2_1']), np.squeeze(return_dict['t1_1']),
                    #                         sliceIdx, net, bvecs_2000[volIdx,:], bvals_2000[volIdx], device)
                    # skio.imsave('dwi_gen_ref.png', gen_slice)
            #         break
            # break
        
        print('')
        
        b0_avrg_vol = np.mean(b0_vols, axis=-1)
        non_b0_mask = [True] * gen_dwi.shape[-1]
        non_b0_mask[:num_b0] = [False]*num_b0
        unscaled = gen_dwi * b0_avrg_vol[..., None]
        gen_dwi[..., non_b0_mask] = unscaled[..., non_b0_mask]
        
        gen_img = nib.Nifti1Image(gen_dwi, dwi_img.affine)
        print(f'/data/s2ms/results/{out_dir}/{subj_id}/data.nii.gz')
        nib.save(gen_img, f'/data/s2ms/results/{out_dir}/{subj_id}/data.nii.gz')
        