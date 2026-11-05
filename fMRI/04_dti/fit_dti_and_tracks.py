from case_config import CASE
from pathlib import Path
import json,time,subprocess,gc
import numpy as np,pandas as pd,nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel,design_matrix
from dipy.data import get_sphere
from dipy.direction.peaks import PeaksAndMetrics
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking import utils
from dipy.tracking.streamline import length,set_number_of_points
from dipy.io.stateful_tractogram import StatefulTractogram,Space
from dipy.io.streamline import save_trk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilearn import plotting
R = CASE.root
O=R/'analysis_rest_dti'
OUT=O/'dti'
P=OUT/'preprocessing'
Q=O/'qc'
REG=OUT/'registration'
REG.mkdir(exist_ok=True)
start=time.time()
CASE.configure_plotting()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':10})
def status(s,**kw):
    (OUT/'status.json').write_text(json.dumps({'status':s,'elapsed_seconds':time.time()-start,**kw},indent=2),encoding='utf-8');print(s,flush=True)
commands=[]
def call(cmd,name):
    cmd=list(map(str,cmd));commands.append(cmd)
    with (REG/name).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
def save(a,name,ref):
    h=ref.header.copy();h.set_data_dtype(np.float32);p=OUT/name;nib.save(nib.Nifti1Image(a.astype(np.float32),ref.affine,h),p);return p
config=json.loads((P/'configuration.json').read_text())
status('TENSOR_FIT')
im=nib.load(P/'eddy_corrected.nii.gz')
data=im.get_fdata(dtype=np.float32)
b=np.loadtxt(CASE.sidecar('dwi','.bval'))
fslvec=np.loadtxt(P/'eddy_corrected.eddy_rotated_bvecs').T
v=fslvec.copy()
handedness=float(np.linalg.det(im.affine[:3,:3]))
if handedness>0:v[:,0]*=-1
assert np.allclose(np.linalg.norm(v[b>50],axis=1),1,atol=1e-4)
original=np.loadtxt(CASE.sidecar('dwi','.bvec')).T
if handedness>0:original[:,0]*=-1
directions=json.loads((O/'work/gradient_dicom_unique.json').read_text())['unique_gradients']
expected=np.array([x['LPS'] for x in directions if x['b']>50])
rot=im.affine[:3,:3]/np.linalg.norm(im.affine[:3,:3],axis=0)
expected=(expected@np.diag([-1,-1,1]))@rot
dot=np.abs(np.sum(expected*original[b>50],axis=1))
assert dot.min()>.9999,('Gradient coordinate mismatch',dot.min())
gtab=gradient_table(b,bvecs=v)
tensor_design=design_matrix(gtab)
assert np.linalg.matrix_rank(tensor_design)==7
mask=nib.load(P/'b0_brain_mask.nii.gz').get_fdata()>.5
S0=data[...,0]
floor=.01*np.median(S0[mask&(S0>0)])
mask&=(S0>floor)&np.isfinite(data).all(axis=3)
save(mask,'tensor_valid_mask.nii.gz',im)
nib.save(nib.Nifti1Image(S0,im.affine),REG/'b0_corrected.nii.gz')
model=TensorModel(gtab,fit_method='WLS')
fit=model.fit(data,mask=mask)
fa=np.nan_to_num(fit.fa)
md=np.nan_to_num(fit.md)
ad=np.nan_to_num(fit.ad)
rd=np.nan_to_num(fit.rd)
assert np.isfinite(fa).all() and fa[mask].min()>=0 and fa[mask].max()<=1+1e-6
for name,a in [('FA',fa),('MD',md),('AD',ad),('RD',rd)]:save(a,f'{name}.nii.gz',im)
pred=fit.predict(gtab,S0=S0)
nrmse=np.sqrt(np.mean((pred-data)**2,axis=3))/np.maximum(S0,1)
nrmse[~mask]=0
save(nrmse,'tensor_normalized_RMSE.nii.gz',im)
del pred
gc.collect()
rgb=(np.abs(fit.evecs[..., :,0])*fa[...,None]*255).astype(np.uint8)
nib.save(nib.Nifti1Image(rgb,im.affine),OUT/'FA_color_voxel_axes.nii.gz')
status('REGISTERING_DWI_TO_T1')
brain=S0*mask
nib.save(nib.Nifti1Image(brain,im.affine),REG/'b0_brain.nii.gz')
t1=R/f'derivatives/fmriprep/{CASE.subject}/anat/{CASE.subject}_desc-preproc_T1w.nii.gz'
t1brain=R/'derivatives/lesion/anatomy/patient_T1_brain.nii.gz'
prefix=REG/'from-DWI_to-T1w_'
warped=REG/'b0_in_T1w.nii.gz'
if not Path(str(prefix)+'0GenericAffine.mat').exists():
    call(['antsRegistration','-d','3','--float','1','-o',f'[{prefix},{warped}]','-n','Linear','--winsorize-image-intensities','[0.005,0.995]',
          '--initial-moving-transform',f'[{t1brain},{REG/"b0_brain.nii.gz"},1]','--transform','Rigid[0.1]',
          '--metric',f'MI[{t1brain},{REG/"b0_brain.nii.gz"},1,32,Regular,0.25]','--convergence','[1000x500x250x100,0.000001,10]',
          '--shrink-factors','8x4x2x1','--smoothing-sigmas','3x2x1x0vox'],'registration.log')
transform=str(prefix)+'0GenericAffine.mat'
inverse=f'[{transform},1]'
def apply(source,reference,dest,interpolation,tx,name):call(['antsApplyTransforms','-d','3','--float','1','-i',source,'-r',reference,'-o',dest,'-n',interpolation,'-t',tx],name)
apply(R/f'derivatives/lesion/anatomy/{CASE.subject}_space-T1w_atlas-JHUWhiteMatter_dseg.nii.gz',REG/'b0_corrected.nii.gz',REG/'JHU_in_DWI.nii.gz','NearestNeighbor',inverse,'jhu_transform.log')
apply(O/'anatomy/motor_ROIs_space-T1w.nii.gz',REG/'b0_corrected.nii.gz',REG/'motor_ROIs_in_DWI.nii.gz','NearestNeighbor',inverse,'motor_transform.log')
apply(R/f'derivatives/fmriprep/{CASE.subject}/anat/{CASE.subject}_label-WM_probseg.nii.gz',REG/'b0_corrected.nii.gz',REG/'WM_probability_in_DWI.nii.gz','Linear',inverse,'wm_transform.log')
apply(OUT/'tensor_valid_mask.nii.gz',t1,REG/'DWI_mask_in_T1w.nii.gz','NearestNeighbor',transform,'brainmask_transform.log')
apply(OUT/'FA.nii.gz',t1,REG/'FA_in_T1w.nii.gz','Linear',transform,'fa_transform.log')
atlas=nib.load(REG/'JHU_in_DWI.nii.gz').get_fdata().astype(int)
motor=nib.load(REG/'motor_ROIs_in_DWI.nii.gz').get_fdata().astype(int)
wm=nib.load(REG/'WM_probability_in_DWI.nii.gz').get_fdata()
bm=nib.load(R/f'derivatives/fmriprep/{CASE.subject}/anat/{CASE.subject}_desc-brain_mask.nii.gz').get_fdata()>.5
dm=nib.load(REG/'DWI_mask_in_T1w.nii.gz').get_fdata()>.5
dice=float(2*(bm&dm).sum()/(bm.sum()+dm.sum()))
assert dice>.65,('Poor global DWI-T1 registration',dice)
figures=[]
for name,a,hi in [('FA',fa,1),('MD',md*1000,2)]:
    image=nib.Nifti1Image(a.astype(np.float32),im.affine)
    disp=plotting.plot_stat_map(image,bg_img=str(REG/'b0_corrected.nii.gz'),display_mode='z',cut_coords=[-20,-5,10,25,40,55,70],threshold=.05 if name=='FA' else .1,vmax=hi,cmap='viridis',title=name+(' (10^-3 mm2/s)' if name=='MD' else '')+' | DWI physical coordinates')
    qp=Q/f'dti_{name}.png';disp.savefig(qp,dpi=150);disp.close();figures.append(str(qp.relative_to(O)))
disp=plotting.plot_anat(str(t1),display_mode='z',cut_coords=[-20,-5,10,25,40,55,70],title='Rigid DWI to patient T1: b0 edges')
disp.add_edges(str(warped))
disp.savefig(Q/'dti_registration_edges.png',dpi=150)
disp.close()
disp=plotting.plot_roi(str(REG/'JHU_in_DWI.nii.gz'),bg_img=str(REG/'b0_corrected.nii.gz'),display_mode='z',cut_coords=[-20,-10,0,10,20,30],alpha=.5,title='JHU reference regions on DWI; not tractography')
disp.savefig(Q/'dti_JHU_reference.png',dpi=150)
disp.close()
jhu_labels=json.loads((R/'derivatives/lesion/anatomy/jhu_labels.json').read_text())
roirows=[]
for lab in [7,8,15,16,19,20]:
    selected=(atlas==lab)&mask&(wm>=.3)
    row={'atlas_label':lab,'region':jhu_labels[str(lab)],'voxel_count':int(selected.sum()),'volume_ml':float(selected.sum()*abs(np.linalg.det(im.affine[:3,:3]))/1000)}
    for name,value in [('FA',fa),('MD',md),('AD',ad),('RD',rd)]:row[name+'_mean']=float(value[selected].mean()) if selected.any() else None
    roirows.append(row)
pd.DataFrame(roirows).to_csv(OUT/'JHU_reference_region_metrics.tsv',sep='\t',index=False)
status('TRACTOGRAPHY')
sphere=get_sphere(name='symmetric362')
tree=cKDTree(sphere.vertices)
vectors=fit.evecs[..., :,0]
_,idx=tree.query(vectors.reshape(-1,3))
idx=idx.reshape(mask.shape).astype(np.int32)
pam=PeaksAndMetrics()
pam.sphere=sphere
pam.peak_indices=idx[...,None]
pam.peak_values=fa[...,None]
pam.qa=fa[...,None]
pam.gfa=fa
pam.ang_thr=30
pam.total_weight=.5
tracks_out={}
trackstats=[]
profiles=[]
trackdir=OUT/'tractography'
trackdir.mkdir(exist_ok=True)
def clip_between_targets(s,pedmask,cortex):
    # Keep only the segment between the two anatomical inclusion targets.
    vox=nib.affines.apply_affine(np.linalg.inv(im.affine),s).T
    pi=np.flatnonzero(ndimage.map_coordinates(pedmask.astype(np.uint8),vox,order=0,mode='constant',cval=0))
    ci=np.flatnonzero(ndimage.map_coordinates(cortex.astype(np.uint8),vox,order=0,mode='constant',cval=0))
    if not len(pi) or not len(ci):return None
    if pi.mean()>ci.mean():return clip_between_targets(s[::-1].copy(),pedmask,cortex)
    after=ci[ci>pi.min()]
    if not len(after):return None
    end=int(after[0]);before=pi[pi<end]
    if not len(before):return None
    return s[int(before[-1]):end+1].copy()
for threshold in [.15,.10]:
    stopmask=mask&(wm>.05);stopfield=fa.copy();stopfield[~stopmask]=0;stopping=ThresholdStoppingCriterion(stopfield,threshold)
    for side,plic,ped,mc,opmc in [('L',20,16,1,2),('R',19,15,2,1)]:
        seedmask=(atlas==plic)&mask&(fa>=threshold);seeds=utils.seeds_from_mask(seedmask,im.affine,density=2)
        candidates=[np.asarray(s) for s in LocalTracking(pam,stopping,seeds,im.affine,.5,maxlen=500,return_all=True,random_seed=20261027)]
        pedmask=ndimage.binary_dilation(atlas==ped,iterations=1);cortex=ndimage.binary_dilation(motor==mc,iterations=2);opp=ndimage.binary_dilation(motor==opmc,iterations=1)
        passing=list(utils.target(candidates,im.affine,pedmask,include=True));passing=list(utils.target(passing,im.affine,cortex,include=True));passing=list(utils.target(passing,im.affine,opp,include=False))
        before_clipping=len(passing)
        clipped=[clip_between_targets(s,pedmask,cortex) for s in passing]
        passing=[s for s in clipped if s is not None and 40<=float(length(s))<=250]
        tag=f'CST_candidate_{side}_FA{threshold:.2f}';st=StatefulTractogram(passing,im,Space.RASMM);save_trk(st,str(trackdir/(tag+'.trk')),bbox_valid_check=True)
        stat={'side':side,'FA_stop_threshold':threshold,'seed_voxels':int(seedmask.sum()),'seeds':len(seeds),'generated_streamlines':len(candidates),'passed_inclusion_before_endpoint_clipping':before_clipping,'qualified_streamlines':len(passing),'status':'RECOVERED_CANDIDATE' if passing else 'NO_QUALIFYING_STREAMLINES_NOT_PROOF_OF_ABSENCE'}
        if passing:
            density=utils.density_map(passing,im.affine,im.shape[:3]);save(density,f'tractography/{tag}_density.nii.gz',im)
            selected=(density>0)&mask&(wm>=.3);stat.update(length_mean_mm=float(np.mean([length(s) for s in passing])),sampled_WM_voxels=int(selected.sum()),FA_mean=float(fa[selected].mean()),MD_mean=float(md[selected].mean()))
            if threshold==.15:
                tracks_out[side]=passing
                apply(trackdir/(tag+'_density.nii.gz'),t1,trackdir/(tag+'_density_T1w.nii.gz'),'Linear',transform,tag+'_toT1.log')
                if len(passing)>=10:
                    ps=[]
                    for s in passing:
                        pts=set_number_of_points(s,100);vox=nib.affines.apply_affine(np.linalg.inv(im.affine),pts);ps.append(ndimage.map_coordinates(fa,vox.T,order=1,mode='nearest'))
                    ps=np.asarray(ps)
                    for node in range(100):profiles.append({'side':side,'normalized_node':node,'FA_median':float(np.median(ps[:,node])),'FA_q25':float(np.percentile(ps[:,node],25)),'FA_q75':float(np.percentile(ps[:,node],75))})
        trackstats.append(stat);print('TRACK',json.dumps(stat),flush=True)
pd.DataFrame(trackstats).to_csv(OUT/'tractography_summary.tsv',sep='\t',index=False)
if profiles:
    table=pd.DataFrame(profiles);table.to_csv(OUT/'along_candidate_tract_FA.tsv',sep='\t',index=False);fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    for side,color in [('L','#dd813d'),('R','#168c9a')]:
        d=table[table.side==side]
        if len(d):ax.plot(d.normalized_node,d.FA_median,color=color,label=side);ax.fill_between(d.normalized_node,d.FA_q25,d.FA_q75,color=color,alpha=.2)
    ax.set(xlabel='候选线条归一化长度：脑干端 → 皮层端',ylabel='FA',ylim=(0,1));ax.legend();ax.grid(alpha=.15);fig.savefig(Q/'dti_candidate_along_FA.png',dpi=155);plt.close(fig)
if tracks_out:
    fig=plt.figure(figsize=(10,8));ax=fig.add_subplot(111,projection='3d')
    for side,color in [('L','#dd813d'),('R','#168c9a')]:
        tr=tracks_out.get(side,[])
        for s in tr[::max(1,len(tr)//400)]:ax.plot(s[:,0],s[:,1],s[:,2],color=color,lw=.5,alpha=.5)
    allpts=np.concatenate([s for tr in tracks_out.values() for s in tr]);span=np.ptp(allpts,axis=0);ax.set_box_aspect(np.maximum(span,1));ax.view_init(15,-90);ax.set(xlabel='R（mm）',ylabel='A（mm）',zlabel='S（mm）',title='DTI约束追踪候选：左橙／右青');fig.tight_layout();fig.savefig(Q/'dti_CST_candidates_3D.png',dpi=155);plt.close(fig)
white=mask&(wm>=.5)
result={'status':'COMPLETE_VISUAL_REVIEW_PENDING','preprocessing':config,'gradient_validation':{'image_axis_codes':nib.aff2axcodes(im.affine),'affine_determinant':handedness,'FSL_to_DTI_xflip':handedness>0,'minimum_absolute_dot_with_ordered_DICOM_voxel_gradients':float(dot.min()),'design_rank':int(np.linalg.matrix_rank(tensor_design)),'design_condition':float(np.linalg.cond(tensor_design))},
 'fit':'DIPY weighted least squares tensor, 1 b0 + 40 b1000 directions','tensor_mask_voxels':int(mask.sum()),'registration_mask_Dice':dice,'WM_FA_median':float(np.median(fa[white])),'WM_MD_median_mm2_s':float(np.median(md[white])),'WM_normalized_fit_RMSE_median':float(np.median(nrmse[white])),'reference_regions':roirows,'tractography':trackstats,
 'tracking_method':{'model':'Single tensor principal direction, quantized to symmetric362 sphere, EuDX direction getter','angle_degrees':30,'step_mm':.5,'seed':'JHU posterior internal capsule, density 2 per axis','inclusion':'ipsilateral cerebral peduncle (1 voxel dilation) and precentral anatomical ROI (2 voxel dilation)','exclusion':'contralateral precentral ROI','endpoint_clipping':'retain segment from last peduncle target contact to first precentral target contact, oriented peduncle to cortex; remove unconstrained terminal extensions','length_mm':[40,250],'primary_FA_stop':.15,'sensitivity_FA_stop':.1,'tract_class':'candidate CST, not hand-specific; no axon count inference'},
 'limitations':['No susceptibility distortion correction; source PE polarity and readout remain unverified beyond internal eddy parameterization','Single b0 and anisotropic 1.75x1.75x3 mm sampling','Population atlas references are approximations, not individually confirmed nuclei or hand-M1','Tensor model cannot resolve crossing fibers; no streamline or low FA directly diagnoses damage','Normalized along-line profiles are descriptive, not exact anatomical node correspondence','No confirmed lesion-side label or healthy control comparison'],
 'commands':commands,'elapsed_seconds':time.time()-start}
(OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
status('COMPLETE',registration_mask_Dice=dice)
print(json.dumps({'Dice':dice,'WM_FA_median':result['WM_FA_median'],'WM_MD_median':result['WM_MD_median_mm2_s'],'tracks':trackstats}),flush=True)
