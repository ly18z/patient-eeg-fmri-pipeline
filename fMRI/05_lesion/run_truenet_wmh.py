"""Offline official paired-modality TrUE-Net MWSC white matter hyperintensity inference."""
from case_config import CASE
import sys, os, json, time
ROOT = CASE.root
ASSETS=CASE.models/'truenet'
sys.path.insert(0,str(ASSETS/'source'))
os.environ['FSLOUTPUTTYPE']='NIFTI_GZ'
import numpy as np,nibabel as nib,torch
from scipy import ndimage
from truenet.true_net import truenet_model,truenet_evaluate,truenet_data_postprocessing
torch.set_num_threads(6)
torch.manual_seed(20261026)
START=time.time()
OUT=ROOT/'derivatives/lesion/model-truenet-wmh'
OUT.mkdir(exist_ok=True)
INPUT=OUT/'input'
INPUT.mkdir(exist_ok=True)
t1=nib.load(ROOT/f'derivatives/fmriprep/{CASE.subject}/anat/{CASE.subject}_desc-preproc_T1w.nii.gz')
flair=nib.load(ROOT/f'derivatives/lesion/{CASE.subject}_space-T1w_desc-coreg_FLAIR.nii.gz')
brain=nib.load(ROOT/f'derivatives/fmriprep/{CASE.subject}/anat/{CASE.subject}_desc-brain_mask.nii.gz').get_fdata()>.5
assert t1.shape==flair.shape==brain.shape and np.allclose(t1.affine,flair.affine)
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def status(phase,**kw):
 write(ROOT/'logs/truenet_status.json',dict(status=phase,elapsed_seconds=time.time()-START,**kw));print(phase,json.dumps(kw),flush=True)
for name,img in [('T1',t1),('FLAIR',flair)]:
 a=img.get_fdata(dtype=np.float32)*brain
 nib.save(nib.Nifti1Image(a,t1.affine),INPUT/f'{CASE.subject}_{name}.nii.gz')
test=dict(basename=f'{CASE.subject}',flair_path=str(INPUT/f'{CASE.subject}_FLAIR.nii.gz'),t1_path=str(INPUT/f'{CASE.subject}_T1.nii.gz'))
planes=[]
weights=[]
for plane in ['axial', 'sagittal', 'coronal']:
    status('INFERENCE_PLANE',plane=plane)
    files=list((ASSETS/'model_data').glob('*FLAIR_T1_'+plane+'.pth'))
    assert len(files)==1,files
    weight=files[0]
    state=torch.load(weight,map_location='cpu',weights_only=True)
    if 'model_state_dict' in state:state=state['model_state_dict']
    state={k.removeprefix('module.'):v for k,v in state.items()}
    model=truenet_model.TrUENet(n_channels=2,n_classes=2,init_channels=64,plane=plane)
    model.load_state_dict(state,strict=True)
    model.eval()
    predicted=truenet_evaluate.evaluate_truenet([test],model,{},torch.device('cpu'),mode=plane,verbose=False)
    restored=truenet_data_postprocessing.resize_to_original_size(predicted,test['flair_path'],plane=plane)
    full=truenet_data_postprocessing.get_final_3dvolumes(restored,test['flair_path'])
    assert full.shape==t1.shape and np.isfinite(full).all() and full.min()>=0 and full.max()<=1
    planes.append(full.astype(np.float32))
    weights.append(dict(file=weight.name))
    nib.save(nib.Nifti1Image(full.astype(np.float32),t1.affine),OUT/f'{CASE.subject}_space-T1w_desc-{plane}WMH_probseg.nii.gz')
    del model,state,predicted,restored
prob=np.mean(planes,axis=0)
mask=(prob>.5)&brain
nib.save(nib.Nifti1Image(prob.astype(np.float32),t1.affine),OUT/f'{CASE.subject}_space-T1w_model-TrUENet_desc-WMH_probseg.nii.gz')
nib.save(nib.Nifti1Image(mask.astype(np.uint8),t1.affine),OUT/f'{CASE.subject}_space-T1w_model-TrUENet_desc-WMHcandidate_mask.nii.gz')
labs,n=ndimage.label(mask,structure=ndimage.generate_binary_structure(3,1))
components=[]
vol=abs(np.linalg.det(t1.affine[:3,:3]))
for j in range(1,n+1):
 ijk=np.argwhere(labs==j);peak=ijk[np.argmax(prob[tuple(ijk.T)])]
 components.append(dict(component_id=j,voxels=len(ijk),volume_ml=float(len(ijk)*vol/1000),
  centroid_native_mm=nib.affines.apply_affine(t1.affine,ijk.mean(axis=0)).tolist(),
  peak_probability=float(prob[tuple(peak)]),mean_probability=float(prob[labs==j].mean())))
components.sort(key=lambda x:x['voxels'],reverse=True)
nib.save(nib.Nifti1Image(labs.astype(np.int16),t1.affine),OUT/f'{CASE.subject}_space-T1w_desc-WMHcomponents_dseg.nii.gz')
result=dict(status='INFERENCE_COMPLETE_REVIEW_PENDING',model='TrUE-Net MWSC paired T1+FLAIR',
 meaning='White matter hyperintensity candidates; not equivalent to stroke infarct/hemorrhage or histological white matter damage',
 preprocessing='N4-corrected T1+FLAIR, rigid registration, T1 brain masking; author Gaussian normalization, tight crop, plane-specific resizing',
 inference='Author three-plane network evaluation and inverse resizing; arithmetic mean of plane probabilities; p>0.5 within brain mask',
 native_grid_verified=True,probability_finite=True,candidate_voxels=int(mask.sum()),candidate_volume_ml=float(mask.sum()*vol/1000),
 max_probability_in_brain=float(prob[brain].max()),components=components,weights=weights,elapsed_seconds=time.time()-START)
write(OUT/'model_result.json',result)
status('INFERENCE_COMPLETE_REVIEW_PENDING',volume_ml=result['candidate_volume_ml'],components=n)
