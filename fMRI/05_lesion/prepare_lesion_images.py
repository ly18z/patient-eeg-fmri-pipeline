"""Register patient's FLAIR to original T1 for chronic-lesion review."""
from case_config import CASE
import json, subprocess, time
import numpy as np,nibabel as nib
import matplotlib
matplotlib.use('Agg')
from nilearn import plotting
ROOT = CASE.root
OUT=ROOT/'derivatives/lesion'
QC=ROOT/'qc/lesion'
LOG=ROOT/'logs'
for p in [OUT,QC,OUT/'registration']:p.mkdir(exist_ok=True,parents=True)
t1=ROOT/f'derivatives/fmriprep/{CASE.subject}/anat/{CASE.subject}_desc-preproc_T1w.nii.gz'
flair=ROOT/f'bids/{CASE.subject}/anat/{CASE.subject}_FLAIR.nii.gz'
corrected=OUT/f'registration/{CASE.subject}_desc-N4_FLAIR.nii.gz'
warped=OUT/f'{CASE.subject}_space-T1w_desc-coreg_FLAIR.nii.gz'
start=time.time()
commands=[]
def call(cmd):
 print(' '.join(map(str,cmd)),flush=True);commands.append(list(map(str,cmd)))
 subprocess.run(list(map(str,cmd)),check=True)
if not corrected.exists():call(['N4BiasFieldCorrection','-d','3','-i',flair,'-o',corrected,'-s','3','-c','[50x50x30,0.00001]'])
if not warped.exists():
 prefix=OUT/f'registration/{CASE.subject}_from-FLAIR_to-T1w_'
 call(['antsRegistration','--dimensionality','3','--float','1','--output',f'[{prefix},{warped}]',
  '--interpolation','Linear','--winsorize-image-intensities','[0.005,0.995]','--use-histogram-matching','0',
  '--initial-moving-transform',f'[{t1},{corrected},1]','--transform','Rigid[0.1]',
  '--metric',f'MI[{t1},{corrected},1,32,Regular,0.25]','--convergence','[1000x500x250x100,0.000001,10]',
  '--shrink-factors','8x4x2x1','--smoothing-sigmas','3x2x1x0vox'])
im=nib.load(warped)
ref=nib.load(t1)
assert im.shape==ref.shape and np.allclose(im.affine,ref.affine)
assert np.isfinite(im.get_fdata()).all()
for modality,p in [('T1',t1),('FLAIR',warped)]:
 display=plotting.plot_anat(str(p),display_mode='z',cut_coords=[-10,0,10,20,30,40,50,60,70],
  title=f'Patient {modality}, original T1 physical space',black_bg=True)
 display.savefig(str(QC/f'{modality}_native_montage.png'),dpi=160);display.close()
with (OUT/'registration/preparation.json').open('w',encoding='utf-8') as f:
 json.dump(dict(status='REGISTERED_VISUAL_REVIEW_PENDING',commands=commands,reference=str(t1.relative_to(ROOT)),
  flair_source=str(flair.relative_to(ROOT)),elapsed_seconds=time.time()-start),f,indent=2)
print('FLAIR_REGISTERED_TO_PATIENT_T1',flush=True)
