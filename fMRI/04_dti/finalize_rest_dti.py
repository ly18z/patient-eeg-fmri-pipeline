from case_config import CASE
from nilearn import plotting
R = CASE.root
O=R/'analysis_rest_dti'
t1=R/f'derivatives/fmriprep/{CASE.subject}/anat/{CASE.subject}_desc-preproc_T1w.nii.gz'
d=plotting.plot_anat(str(t1),display_mode='z',cut_coords=[-20,-5,10,25,40,55,70],title='Candidate CST on patient T1: left orange / right cyan')
for side,color in [('L','#dd813d'),('R','#168c9a')]:
    p=O/f'dti/tractography/CST_candidate_{side}_FA0.15_density_T1w.nii.gz'
    if p.exists():d.add_contours(str(p),levels=[.1],colors=[color],linewidths=.8)
d.savefig(O/'qc/dti_CST_on_T1.png',dpi=160)
d.close()
print('SUPPLEMENTAL_QC_READY')
