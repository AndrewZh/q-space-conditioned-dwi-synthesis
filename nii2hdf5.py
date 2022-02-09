import numpy as np
import h5py
import nibabel as nib
import os
import os.path as osp
from dipy.io import read_bvals_bvecs
from dipy.reconst.shm import normalize_data

def preprocess_struct_MRI(mri):
    # mri /= mri.max() 
    for z in range(mri.shape[2]):
        mri[...,z] = mri[...,z] / (1e-6 + mri[...,z].max())
    return mri

def preprocess_DWI(mri, b0_mask):
    dwi_mask = np.logical_not(b0_mask)
    dwi_vols = mri[..., dwi_mask]

    norm_mri = normalize_data(mri, b0_mask)
    norm_dwi = norm_mri[..., dwi_mask]
    norm_dwi[dwi_vols==0] = 0
    norm_dwi = np.clip(norm_dwi, 0, 1)

    b0_vols = mri[..., b0_mask]
    b0_avrg_vol = np.mean(b0_vols, axis=-1)
    b0_max = b0_avrg_vol.max()
    b0_avrg_vol = b0_avrg_vol / b0_max
    b0_avrg_vol[b0_avrg_vol < 0] = 0

    return b0_avrg_vol, norm_dwi


def nii2hdf5(hcp_dir, subjects, target_mode):    
    # dwi_data = None
    # bval_bvec_data = None
    # t1_data = None
    # t2_data = None
    # b0_data = None
    
    num_subjects = len(subjects)
    with h5py.File(osp.join(hcp_dir, f'{target_mode}_{num_subjects}_nonPad_NoStructNorm_4Qsampl.hdf5'), 'w') as data_file:                  
        for i, subject in enumerate(subjects):
            print(f'\n{subject}')
            brain_mask = nib.load(osp.join(hcp_dir, subject, 'nodif_brain_mask.nii.gz')).get_fdata()
            t1 = nib.load(osp.join(hcp_dir, subject, f'{subject}_t1_125.nii.gz')).get_fdata()
            t1[brain_mask==0] = 0
            t2 = nib.load(osp.join(hcp_dir, subject, f'{subject}_t2_125.nii.gz')).get_fdata()
            t2[brain_mask==0] = 0
            dwi = nib.load(osp.join(hcp_dir, subject, 'data.nii.gz')).get_fdata()
            dwi[brain_mask==0,:] = 0 
            bval, bvec = read_bvals_bvecs(osp.join(hcp_dir, subject, 'bvals'),
                                        osp.join(hcp_dir, subject, 'bvecs'))
            bval[bval < 10] = 0
            
            bval_bvec = np.concatenate((bvec, np.expand_dims(bval,axis=1)),axis=1)

            b0_mask = (bval == 0)
            bval_bvec = bval_bvec[np.logical_not(b0_mask),:]
            
            b0_avrg_vol, norm_dwi = preprocess_DWI(dwi, b0_mask)
            # t1 = preprocess_struct_MRI(t1)
            # t2 = preprocess_struct_MRI(t2)

            flat_slices = np.reshape(brain_mask,(-1, brain_mask.shape[-1]))
            slice_mask = np.sum(flat_slices, axis=0) > 0
            for vol_idx in range(norm_dwi.shape[-1]):
                print('{}:\t{} / {}'.format(subject, vol_idx, norm_dwi.shape[-1]), end='\r')
                
                b0_vol = b0_avrg_vol
                dwi_vol = norm_dwi[..., vol_idx]
                
                vol_bval_bvecs = np.repeat(np.expand_dims(bval_bvec[vol_idx,:], axis=0),
                                            repeats=norm_dwi.shape[2], axis=0)
                dwi_vol = np.moveaxis(dwi_vol, 2, 0)
                t1_vol = np.moveaxis(t1, 2, 0)
                t2_vol = np.moveaxis(t2, 2, 0)
                b0_vol = np.moveaxis(b0_vol, 2, 0)
                
                vol_bval_bvecs = vol_bval_bvecs[slice_mask,...]
                dwi_vol = dwi_vol[slice_mask, ...]
                t1_vol = t1_vol[slice_mask, ...]
                t2_vol = t2_vol[slice_mask, ...]
                b0_vol = b0_vol[slice_mask, ...]

                t1_vol = t1_vol.astype(np.float32)
                t2_vol = t2_vol.astype(np.float32)
                dwi_vol = dwi_vol.astype(np.float32)
                vol_bval_bvecs = vol_bval_bvecs.astype(np.float32)
                b0_vol = b0_vol.astype(np.float32)

                if f'{target_mode}_t1' not in data_file:
                    data_file.create_dataset(f'{target_mode}_t1', data=t1_vol,
                                                        maxshape=(None, t1_vol.shape[1], t1_vol.shape[2]))
                    data_file.create_dataset(f'{target_mode}_t2', data=t2_vol,
                                                        maxshape=(None, t2_vol.shape[1], t2_vol.shape[2]))
                    data_file.create_dataset(f'{target_mode}_dwi', data=dwi_vol,
                                                        maxshape=(None, dwi_vol.shape[1], dwi_vol.shape[2]))
                    data_file.create_dataset(f'{target_mode}_bval_vec', data=vol_bval_bvecs,
                                                                maxshape=(None, vol_bval_bvecs.shape[1]))
                    data_file.create_dataset(f'{target_mode}_b0', data=b0_vol, 
                                                        maxshape=(None, b0_vol.shape[1], b0_vol.shape[2]))
                else:
                    assert data_file[f'{target_mode}_t1'].shape[0] == data_file[f'{target_mode}_t2'].shape[0]
                    assert data_file[f'{target_mode}_t1'].shape[0] == data_file[f'{target_mode}_dwi'].shape[0]
                    assert data_file[f'{target_mode}_t1'].shape[0] == data_file[f'{target_mode}_bval_vec'].shape[0]
                    assert data_file[f'{target_mode}_t1'].shape[0] == data_file[f'{target_mode}_b0'].shape[0]
                    
                    current_slice_num = data_file[f'{target_mode}_t1'].shape[0]
                    data_file[f'{target_mode}_t1'].resize((current_slice_num+t1_vol.shape[0], t1_vol.shape[1], t1_vol.shape[2]))
                    data_file[f'{target_mode}_t1'][-t1_vol.shape[0]:,:,:] = t1_vol

                    data_file[f'{target_mode}_t2'].resize((current_slice_num+t2_vol.shape[0], t2_vol.shape[1], t2_vol.shape[2]))
                    data_file[f'{target_mode}_t2'][-t2_vol.shape[0]:,:,:] = t2_vol

                    data_file[f'{target_mode}_dwi'].resize((current_slice_num+dwi_vol.shape[0], dwi_vol.shape[1], dwi_vol.shape[2]))
                    data_file[f'{target_mode}_dwi'][-dwi_vol.shape[0]:,:,:] = dwi_vol

                    data_file[f'{target_mode}_bval_vec'].resize((current_slice_num+vol_bval_bvecs.shape[0], vol_bval_bvecs.shape[1]))
                    data_file[f'{target_mode}_bval_vec'][-vol_bval_bvecs.shape[0]:,:] = vol_bval_bvecs

                    data_file[f'{target_mode}_b0'].resize((current_slice_num+b0_vol.shape[0], b0_vol.shape[1], b0_vol.shape[2]))
                    data_file[f'{target_mode}_b0'][-b0_vol.shape[0]:,:,:] = b0_vol
        

if __name__ == "__main__":
    data_root = '/data/s2ms/' 
    target_mode = 'train'
    
    data_dir = osp.join(data_root, target_mode)

    #TODO: Get IDs from already saved hdf5
    subjects = ['601127', '622236', '644044', '645551', 
                '654754', '704238', '715041', '748258', '761957',
                '784565', '792564', '814649', '837560', '857263', '859671']
    #[s for s in os.listdir(data_dir) if osp.isdir(osp.join(data_dir, s))]
    nii2hdf5(data_dir, subjects, target_mode)

    # target_mode = 'test'
    
    # data_dir = osp.join(data_root, target_mode)

    # #TODO: Get IDs from already saved hdf5
    # subjects = ['872158']#,  '889579',  '894673',  '901442',  '910241',  '912447',  '922854',
    #             # '930449',  '932554',  '958976',  '978578',  '979984',  '983773',  '991267']
    # nii2hdf5(data_dir, subjects, target_mode)
