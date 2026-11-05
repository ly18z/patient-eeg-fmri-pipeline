from case_config import CASE
import json, subprocess, time
import numpy as np,nibabel as nib
from dipy.denoise.localpca import mppca
R = CASE.root
O=R/'analysis_rest_dti'
OUT=O/'dti'
OUT.mkdir(parents=True,exist_ok=True)
P=OUT/'preprocessing'
P.mkdir(parents=True,exist_ok=True)
start=time.time()
def status(s,**kw):
    (OUT/'status.json').write_text(json.dumps({'status':s,'elapsed_seconds':time.time()-start,**kw},indent=2),encoding='utf-8');print(s,flush=True)
def call(cmd,log):
    with (P/log).open('w') as f:subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,check=True)
status('BRAIN_MASK')
src=CASE.input('dwi')
im=nib.load(src)
data=im.get_fdata(dtype=np.float32)
b=np.loadtxt(CASE.sidecar('dwi','.bval'))
v=np.loadtxt(CASE.sidecar('dwi','.bvec'))
assert data.ndim==4 and len(b)==data.shape[3] and v.shape==(3,len(b)) and np.isfinite(data).all()
assert np.allclose(np.linalg.norm(v[:,b>50],axis=0),1,atol=1e-5)
b0=P/'b0.nii.gz'
nib.save(nib.Nifti1Image(data[...,b<50].mean(axis=3),im.affine),b0)
call(['bet2',b0,P/'b0_brain','-m','-f','.25'],'bet.log')
mask=nib.load(P/'b0_brain_mask.nii.gz').get_fdata()>.5
volume=float(mask.sum()*abs(np.linalg.det(im.affine[:3,:3]))/1e6)
den=P/'dwi_desc-MPPCA.nii.gz'
if not den.exists():
    status('MPPCA_DENOISING',mask_volume_L=volume)
    denoised,sigma=mppca(data,mask=mask,patch_radius=2,return_sigma=True)
    assert denoised.shape==data.shape and np.isfinite(denoised).all()
    nib.save(nib.Nifti1Image(denoised.astype(np.float32),im.affine),den);nib.save(nib.Nifti1Image(sigma.astype(np.float32),im.affine),P/'MP_PCA_noise_sigma.nii.gz')
else:sigma=nib.load(P/'MP_PCA_noise_sigma.nii.gz').get_fdata()
meta=json.loads((CASE.sidecar('dwi','.json')).read_text())
direction=meta.get('PhaseEncodingDirection') or meta['PhaseEncodingAxis']
axis=direction[0]
sign=-1 if direction.endswith('-') else 1
encoding=np.eye(3,dtype=int)['ijk'.index(axis)]*sign
readout=float(meta.get('TotalReadoutTime') or meta['EstimatedTotalReadoutTime'])
assert .005<readout<.5
(P/'acqparams.txt').write_text(' '.join(map(str,encoding))+f' {readout:.8f}\n')
(P/'index.txt').write_text(' '.join(['1']*len(b))+'\n')
eddy=CASE.tools/'fsl_eddy/bin/eddy_cpu'
cmd=[eddy,f'--imain={den}',f'--mask={P/"b0_brain_mask.nii.gz"}',f'--acqp={P/"acqparams.txt"}',f'--index={P/"index.txt"}',f'--bvecs={CASE.sidecar("dwi",".bvec")}',f'--bvals={CASE.sidecar("dwi",".bval")}',f'--out={P/"eddy_corrected"}','--data_is_shelled','--repol','--ol_type=sw','--ol_nstd=4','--niter=5','--fwhm=10,6,4,2,0','--nthr=4','--initrand=20261027','--cnr_maps','--verbose']
record={'source':str(src.relative_to(R)),'mask_volume_L':volume,'MP_PCA_patch_radius':2,'median_MP_PCA_sigma_in_mask':float(np.median(sigma[mask])),'eddy_command':list(map(str,cmd)),
 'eddy_executable':str(eddy),'topup':False,'susceptibility_distortion_correction':False,
 'acquisition_parameters':{'PE_axis':axis,'PE_sign':direction if 'PhaseEncodingDirection' in meta else 'Positive axis used as internal convention; acquisition polarity unverified','readout_seconds':readout,'readout_source':'TotalReadoutTime' if 'TotalReadoutTime' in meta else 'dcm2niix estimate, not independently verified','rationale':'Single PE axis/time; eddy run without external field/topup; encoding sign and field scale are internal conventions per FSL eddy FAQ. No claim of susceptibility correction.'},
 'Gibbs_removal':False,'Gibbs_reason':'Gibbs removal is not enabled in this pipeline','status':'EDDY_RUNNING'}
(P/'configuration.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
if not (P/'eddy_corrected.eddy_rotated_bvecs').exists():
    status('EDDY_RUNNING');call(cmd,'eddy.log')
corrected=nib.load(P/'eddy_corrected.nii.gz')
rot=np.loadtxt(P/'eddy_corrected.eddy_rotated_bvecs')
assert corrected.shape==im.shape and np.allclose(corrected.affine,im.affine) and rot.shape==v.shape
record['status']='COMPLETE'
record['elapsed_seconds']=time.time()-start
if (P/'eddy_corrected.eddy_movement_rms').exists():
    rms=np.loadtxt(P/'eddy_corrected.eddy_movement_rms');record['movement_RMS_mm']={'absolute_mean':float(rms[:,0].mean()),'absolute_max':float(rms[:,0].max()),'relative_mean':float(rms[1:,1].mean()),'relative_max':float(rms[:,1].max())}
if (P/'eddy_corrected.eddy_outlier_map').exists():
    outliers=np.loadtxt(P/'eddy_corrected.eddy_outlier_map',skiprows=1);record['outlier_slices']={'count':int((outliers>0).sum()),'total':int(outliers.size),'fraction':float((outliers>0).mean())}
(P/'configuration.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
status('PREPROCESSING_COMPLETE',elapsed_seconds=time.time()-start)
print(json.dumps(record),flush=True)
