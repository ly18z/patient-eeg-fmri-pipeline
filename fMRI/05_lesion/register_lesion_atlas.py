"""Register the Harvard-Oxford atlas's own MNI6 template to native patient T1.

MNI6 is not assumed to equal the prior GLM's MNI2009 template. Labels are a
warped population atlas, not a direct patient-specific nuclear segmentation.
"""
from case_config import CASE
from pathlib import Path
import json, subprocess, time, xml.etree.ElementTree as ET
import numpy as np,nibabel as nib
import matplotlib
matplotlib.use('Agg')
from nilearn import plotting
ROOT = CASE.root
START=time.time()
ATLAS=ROOT/'work/lesion_atlas'
OUT=ROOT/'derivatives/lesion/anatomy'
OUT.mkdir(parents=True,exist_ok=True)
QC=ROOT/'qc/lesion'
QC.mkdir(parents=True,exist_ok=True)
T1=ROOT/f'derivatives/fmriprep/{CASE.subject}/anat/{CASE.subject}_desc-preproc_T1w.nii.gz'
MASK=ROOT/f'derivatives/fmriprep/{CASE.subject}/anat/{CASE.subject}_desc-brain_mask.nii.gz'
temp=nib.load(ATLAS/'tpl-MNI152NLin6Asym_res-02_T1w.nii.gz')
tm=nib.load(ATLAS/'tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz').get_fdata()>.5
ref=nib.load(T1)
rm=nib.load(MASK).get_fdata()>.5
for arr,img,p in [(temp.get_fdata(dtype=np.float32)*tm,temp,OUT/'MNI6_brain.nii.gz'),(ref.get_fdata(dtype=np.float32)*rm,ref,OUT/'patient_T1_brain.nii.gz')]:
 nib.save(nib.Nifti1Image(arr,img.affine),p)
fixed=OUT/'patient_T1_brain.nii.gz'
moving=OUT/'MNI6_brain.nii.gz'
prefix=OUT/'from-MNI6_to-T1w_'
warped=OUT/'MNI6_in_patient_T1.nii.gz'
commands=[]
def call(cmd):
 commands.append(list(map(str,cmd)));print(' '.join(map(str,cmd)),flush=True);subprocess.run(list(map(str,cmd)),check=True)
if not Path(str(prefix)+'1Warp.nii.gz').exists():
 call(['antsRegistration','-d','3','--float','1','-o',f'[{prefix},{warped}]','-n','Linear',
  '--winsorize-image-intensities','[0.005,0.995]','--use-histogram-matching','0',
  '--initial-moving-transform',f'[{fixed},{moving},1]',
  '--transform','Affine[0.1]','--metric',f'MI[{fixed},{moving},1,32,Regular,0.25]',
  '--convergence','[500x250x100,0.000001,10]','--shrink-factors','4x2x1','--smoothing-sigmas','2x1x0vox',
  '--transform','SyN[0.1,3,0]','--metric',f'CC[{fixed},{moving},1,3]',
  '--convergence','[60x30x10,0.000001,10]','--shrink-factors','4x2x1','--smoothing-sigmas','2x1x0vox'])
mapped=OUT/f'{CASE.subject}_space-T1w_atlas-HarvardOxfordSubcortical_dseg.nii.gz'
call(['antsApplyTransforms','-d','3','--float','1','-i',ATLAS/'HarvardOxford-sub-maxprob-thr25-1mm.nii.gz',
 '-r',T1,'-o',mapped,'-n','NearestNeighbor','-t',str(prefix)+'1Warp.nii.gz','-t',str(prefix)+'0GenericAffine.mat'])
im=nib.load(mapped)
data=im.get_fdata()
assert im.shape==ref.shape and np.allclose(im.affine,ref.affine)
lut_ids=[2,3,4,10,11,12,13,16,17,18,26,41,42,43,49,50,51,52,53,54,58]
labels={lut_ids[int(e.attrib['index'])]:e.text.strip() for e in ET.parse(ATLAS/'HarvardOxford-Subcortical.xml').findall('.//label')}
(OUT/'atlas_labels.json').write_text(json.dumps(labels,indent=2),encoding='utf-8')
for name,ids in [('basalGanglia',[11,12,13,26,50,51,52,58]),('whiteMatter',[2,41])]:
 roi=np.isin(data,ids).astype(np.uint8);out=OUT/f'{CASE.subject}_space-T1w_atlas-HO_desc-{name}_mask.nii.gz'
 nib.save(nib.Nifti1Image(roi,ref.affine),out)
 disp=plotting.plot_roi(str(out),bg_img=str(T1),display_mode='z',cut_coords=[0,5,10,15,20,25,30,35],
  title=f'Patient native T1 | warped Harvard-Oxford {name}',alpha=.35)
 disp.savefig(str(QC/f'atlas_{name}_overlay.png'),dpi=160);disp.close()
recovered=nib.load(warped).get_fdata()
valid=rm&(recovered>0)
corr=float(np.corrcoef(ref.get_fdata()[valid],recovered[valid])[0,1])
assert corr>.5,corr
result=dict(status='REGISTERED_REVIEW_PENDING',atlas='Harvard-Oxford subcortical maxprob thr25 1mm',
 source_template='MNI152NLin6Asym',target='native T1',native_grid_verified=True,label_encoding='FreeSurfer aseg IDs in republished HO atlas; XML order explicitly decoded',
 registered_template_intensity_correlation=corr,labels=labels,commands=commands,elapsed_seconds=time.time()-START,
 limitation='Population-atlas approximation; overlaps are localization estimates, not patient-specific nuclear boundary truth')
(OUT/'atlas_registration.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('ATLAS_REGISTERED',corr,flush=True)
