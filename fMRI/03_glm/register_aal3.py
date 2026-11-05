from case_config import CASE
from pathlib import Path
import json, time, subprocess, xml.etree.ElementTree as ET
import numpy as np,nibabel as nib
from nibabel.processing import resample_from_to
from nilearn import plotting
R = CASE.root
O=R/'analysis_glm_regions'
S=O/'atlas/AAL3_source'
D=O/'atlas/AAL3_registration'
D.mkdir(exist_ok=True)
T=R/f'derivatives/fmriprep/{CASE.subject}/anat'
start=time.time()
commands=[]
def call(cmd,name):
 cmd=list(map(str,cmd));commands.append(cmd)
 with (D/name).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
source=S/'AAL3v1_1mm.nii'
im=nib.load(source)
a=np.rint(im.get_fdata()).astype(int)
ids={int(x.find('index').text):x.find('name').text for x in ET.parse(S/'AAL3v1_1mm.xml').findall('.//label')}
actual=set(np.unique(a))-{0}
temp=nib.load(S/'tpl-MNIColin27_T1w.nii.gz')
mask=nib.load(S/'tpl-MNIColin27_desc-brain_mask.nii.gz')
tm=resample_from_to(mask,temp,order=0).get_fdata()>.5
nib.save(nib.Nifti1Image((temp.get_fdata()*tm).astype(np.float32),temp.affine),D/'Colin27_brain.nii.gz')
source_coverage=float((resample_from_to(mask,im,order=0).get_fdata()[a>0]>.5).mean())
assert source_coverage>.9,('AAL3/Colin27 template mismatch',source_coverage)
fixed=R/'derivatives/lesion/anatomy/patient_T1_brain.nii.gz'
moving=D/'Colin27_brain.nii.gz'
prefix=D/'from-Colin27_to-T1w_'
warped=D/'Colin27_in_patient_T1.nii.gz'
if not Path(str(prefix)+'1Warp.nii.gz').exists():
 print('REGISTERING_COLIN27_TO_PATIENT_T1',flush=True)
 call(['antsRegistration','-d','3','--float','1','-o',f'[{prefix},{warped}]','-n','Linear','--winsorize-image-intensities','[0.005,0.995]','--use-histogram-matching','0','--initial-moving-transform',f'[{fixed},{moving},1]',
  '--transform','Affine[0.1]','--metric',f'MI[{fixed},{moving},1,32,Regular,0.25]','--convergence','[500x250x100,0.000001,10]','--shrink-factors','4x2x1','--smoothing-sigmas','2x1x0vox',
  '--transform','SyN[0.1,3,0]','--metric',f'CC[{fixed},{moving},1,3]','--convergence','[60x30x10,0.000001,10]','--shrink-factors','4x2x1','--smoothing-sigmas','2x1x0vox'],'registration.log')
ref=R/f'derivatives/glm/{CASE.subject}_task-{CASE.task}_combined/{CASE.subject}_task-{CASE.task}_combined_contrast-taskVsRest_desc-zmap.nii.gz'
dest=O/'atlas/AAL3_space-MNI152NLin2009cAsym_res-2_dseg.nii.gz'
base=['antsApplyTransforms','-d','3','--float','1','-i',source,'-r',ref,'-o',dest,'-n','NearestNeighbor','-t',T/f'{CASE.subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5','-t',str(prefix)+'1Warp.nii.gz','-t',str(prefix)+'0GenericAffine.mat']
call(base,'aal3_transform.log')
native=D/'AAL3_in_patient_T1.nii.gz'
call(['antsApplyTransforms','-d','3','--float','1','-i',source,'-r',fixed,'-o',native,'-n','NearestNeighbor','-t',str(prefix)+'1Warp.nii.gz','-t',str(prefix)+'0GenericAffine.mat'],'aal3_native.log')
call(['antsApplyTransforms','-d','3','--float','1','-i',S/'tpl-MNIColin27_desc-brain_mask.nii.gz','-r',fixed,'-o',D/'Colin27_mask_in_T1w.nii.gz','-n','NearestNeighbor','-t',str(prefix)+'1Warp.nii.gz','-t',str(prefix)+'0GenericAffine.mat'],'mask_transform.log')
f=nib.load(fixed).get_fdata()
w=nib.load(warped).get_fdata()
fm=nib.load(T/f'{CASE.subject}_desc-brain_mask.nii.gz').get_fdata()>.5
wm=nib.load(D/'Colin27_mask_in_T1w.nii.gz').get_fdata()>.5
valid=fm&(w>0)
corr=float(np.corrcoef(f[valid],w[valid])[0,1])
dice=float(2*(fm&wm).sum()/(fm.sum()+wm.sum()))
assert corr>.5 and dice>.8,(corr,dice)
out=nib.load(dest)
data=np.rint(out.get_fdata()).astype(int)
reference=nib.load(ref)
assert out.shape==reference.shape and np.allclose(out.affine,reference.affine)
assert set(np.unique(data))<=({0}|actual)
laterality={}
for side,sgn in [('L',-1),('R',1)]:
 keep=[i for i,n in ids.items() if n.endswith('_'+side)];xyz=nib.affines.apply_affine(out.affine,np.argwhere(np.isin(data,keep)));frac=float((sgn*xyz[:,0]>=0).mean());assert frac>.95;laterality[side]=frac
for mode,img,bg,cuts in [('source',source,S/'tpl-MNIColin27_T1w.nii.gz',[-45,-25,-5,15,35,55,75]),('MNI2009',dest,T/f'{CASE.subject}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz',[-45,-25,-5,15,35,55,75])]:
 disp=plotting.plot_roi(str(img),bg_img=str(bg),display_mode='z',cut_coords=cuts,alpha=.5,colorbar=False,title='AAL3 on '+mode);disp.savefig(O/f'qc/atlas_AAL3_{mode}.png',dpi=150);disp.close()
disp=plotting.plot_anat(str(fixed),display_mode='z',cut_coords=[-45,-25,-5,15,35,55,75],title='Colin27 to patient T1: registered template edges')
disp.add_edges(str(warped))
disp.savefig(O/'qc/AAL3_registration_edges.png',dpi=150)
disp.close()
result = {'status': 'REGISTERED_INITIAL_VISUAL_REVIEW_PENDING', 'release': 'AAL3v2 package (April2024); atlas volume remains AAL3v1 as stated by publisher', 'source_template': 'MNIColin27', 'source_labels': len(actual), 'max_label': 170, 'empty_original_ids': [35, 36, 81, 82], 'source_nonzero_fraction_within_Colin27_brain_mask': source_coverage, 'target_space': 'MNI152NLin2009cAsym, exact existing GLM grid', 'mapping': 'Colin27 -> patient native T1 (Affine + SyN), then existing fMRIPrep T1 -> MNI2009; composed transforms and one nearest-neighbor resampling for labels', 'registration_intensity_correlation': corr, 'registration_brain_mask_Dice': dice, 'target_laterality_fraction': laterality, 'labels_retained_on_target': len(set(np.unique(data)) - {0}), 'labels_lost_on_target': [{'id': i, 'name': ids[i]} for i in sorted(actual - set(np.unique(data)))], 'commands': commands, 'source_files': [{'file': str(p.relative_to(R))} for p in [source, S / 'AAL3v1_1mm.xml', S / 'tpl-MNIColin27_T1w.nii.gz', S / 'tpl-MNIColin27_desc-brain_mask.nii.gz']], 'elapsed_seconds': time.time() - start}
for row in result['labels_lost_on_target']:row['id']=int(row['id'])
(O/'atlas/AAL3_registration.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
