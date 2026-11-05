from case_config import CASE
import json, subprocess
import numpy as np,nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib
matplotlib.use('Agg')
from nilearn import plotting
R = CASE.root
O=R/'analysis_rest_dti'
A=O/'anatomy'
Q=O/'qc'
A.mkdir(parents=True,exist_ok=True)
Q.mkdir(parents=True,exist_ok=True)
T=R/f'derivatives/fmriprep/{CASE.subject}/anat'
L=R/'derivatives/lesion/anatomy'
t1=T/f'{CASE.subject}_desc-preproc_T1w.nii.gz'
gm=T/f'{CASE.subject}_label-GM_probseg.nii.gz'
brain=T/f'{CASE.subject}_desc-brain_mask.nii.gz'
src=O/'work/atlas/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz'
im=nib.load(src)
d=im.get_fdata()
assert set(np.unique(d))<=set(range(49))
world_x=np.indices(im.shape)[0]*im.affine[0,0]+im.affine[0,3]
roi=np.zeros(im.shape,dtype=np.uint8)
labels={}
rid=0
names={6:('中央前回参考区','Precentral'),25:('辅助运动区参考','SMA'),16:('中央后回参考区','Postcentral')}
for atlas_index,(cn,en) in names.items():
    for side in ['L','R']:
        rid+=1;roi[(d==atlas_index+1)&((world_x<0) if side=='L' else (world_x>=0))]=rid
        labels[rid]={'id':rid,'name':('左' if side=='L' else '右')+cn,'short':side+' '+en,'hemisphere':side,'source_index':atlas_index+1}
source=A/'motor_ROIs_space-MNI6.nii.gz'
nib.save(nib.Nifti1Image(roi,im.affine),source)
native_raw=A/'motor_ROIs_space-T1w_unrestricted.nii.gz'
native=A/'motor_ROIs_space-T1w.nii.gz'
standard=A/'motor_ROIs_space-MNI2009_res-2.nii.gz'
commands=[]
def call(cmd):
    commands.append(list(map(str,cmd)));print('ROI_TRANSFORM',flush=True);subprocess.run(list(map(str,cmd)),check=True)
call(['antsApplyTransforms','-d','3','--float','1','-i',source,'-r',t1,'-o',native_raw,'-n','NearestNeighbor','-t',L/'from-MNI6_to-T1w_1Warp.nii.gz','-t',L/'from-MNI6_to-T1w_0GenericAffine.mat'])
n=nib.load(native_raw)
nd=n.get_fdata().astype(np.uint8)
g=nib.load(gm).get_fdata()
bm=nib.load(brain).get_fdata()>.5
assert g.shape==nd.shape
nd[(g<.2)|~bm]=0
nib.save(nib.Nifti1Image(nd,n.affine),native)
if 'rest' not in CASE.data['inputs']:
    (A/'motor_ROIs.json').write_text(json.dumps({'atlas':'HarvardOxford cortical maxprob thr25 1mm',
        'labels':labels,'coverage':[],'commands':commands,
        'scope':'Native anatomical ROIs for DTI; no resting BOLD supplied'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Native motor ROIs ready for DTI')
else:
    rest_ref=R/f'derivatives/fmriprep/{CASE.subject}/func/{CASE.subject}_task-rest_space-MNI152NLin2009cAsym_res-2_boldref.nii.gz'
    call(['antsApplyTransforms','-d','3','--float','1','-i',native,'-r',rest_ref,'-o',standard,'-n','NearestNeighbor','-t',T/f'{CASE.subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5'])
    restmask=nib.load(R/f'derivatives/fmriprep/{CASE.subject}/func/{CASE.subject}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz')
    sm=nib.load(standard).get_fdata().astype(int)
    m=restmask.get_fdata()>.5
    native_mask=nib.load(R/f'derivatives/fmriprep/{CASE.subject}/func/{CASE.subject}_task-rest_space-T1w_desc-brain_mask.nii.gz')
    nm=resample_from_to(native_mask,n,order=0).get_fdata()>.5
    coverage=[]
    for k,label in labels.items():
        full=sm==k;nn=nd==k;frac=float((full&m).sum()/max(1,full.sum()));frac_native=float((nn&nm).sum()/max(1,nn.sum()))
        assert (full&m).sum()>=20,('Insufficient ROI',label)
        coverage.append({**label,'native_roi_voxels':int(nn.sum()),'native_BOLD_coverage_fraction':frac_native,'standard_roi_voxels':int(full.sum()),'standard_valid_voxels':int((full&m).sum()),'standard_coverage_fraction':frac,'coverage_flag':'PASS' if min(frac,frac_native)>=.8 else 'LIMITED_COVERAGE'})
    mask_sm=sm.copy()
    mask_sm[~m]=0
    nib.save(nib.Nifti1Image(mask_sm.astype(np.uint8),restmask.affine),A/'motor_ROIs_space-MNI2009_desc-restCovered_dseg.nii.gz')
    bg=T/f'{CASE.subject}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz'
    disp=plotting.plot_roi(str(standard),bg_img=str(bg),display_mode='z',cut_coords=[30,40,50,60,70],alpha=.6,colorbar=True,title='Anatomical motor ROIs: independent of task activation')
    disp.savefig(Q/'motor_ROIs_MNI.png',dpi=150)
    disp.close()
    for name,img,back in [('native',native,t1),('standard',standard,bg)]:
        disp=plotting.plot_roi(str(img),bg_img=str(back),display_mode='ortho',cut_coords=(0,-20,55),alpha=.5,title='Motor ROIs: '+name)
        disp.savefig(Q/f'motor_ROIs_{name}_ortho.png',dpi=150);disp.close()
    (A/'motor_ROIs.json').write_text(json.dumps({'status':'READY_VISUAL_QC_PENDING','atlas':'HarvardOxford cortical maxprob thr25 1mm; hemisphere split in source MNI6','GM_probability_min':.2,'labels':labels,'coverage':coverage,'commands':commands,'limitation':'Precentral anatomical ROI is not a hand-specific or individually confirmed M1 subdivision; no ROI selected by task significance.'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(coverage,ensure_ascii=False),flush=True)
