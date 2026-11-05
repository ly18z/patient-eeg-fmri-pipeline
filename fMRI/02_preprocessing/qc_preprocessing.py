"""Verify completed fMRIPrep outputs and summarize motion; no GLM/denoising."""
from case_config import CASE
import sys, json, csv, gc, re
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root = CASE.root
motion_only='--motion-only' in sys.argv
out=root/'derivatives/fmriprep'
qc=root/'qc/preprocessing'
qc.mkdir(parents=True,exist_ok=True)
func=out/f'{CASE.subject}/func'
anat=out/f'{CASE.subject}/anat'
sources=sorted((root/f'bids/{CASE.subject}/func').glob('*_bold.nii.gz'))
records=[]
plot_records=[]
def run_key(name):
 task=re.search(r'_task-([^_]+)',name).group(1)
 run=re.search(r'_run-(\d+)',name)
 return task,int(run.group(1)) if run else 0
for src in sources:
    source_stem=src.name.removesuffix('_bold.nii.gz')
    raw=nib.load(src)
    n=raw.shape[3]
    matches=[p for p in func.glob('*_desc-confounds_timeseries.tsv') if run_key(p.name)==run_key(src.name)]
    conf=matches[0]
    stem=conf.name.removesuffix('_desc-confounds_timeseries.tsv')
    with conf.open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
    assert len(rows)==n,(stem,len(rows),n)
    def col(k):
     return np.array([float(r[k]) if r.get(k) not in [None,'n/a','','NaN'] else np.nan for r in rows])
    fd=col('framewise_displacement')
    dv=col('std_dvars')
    nonsteady=[k for k in rows[0] if k.startswith('non_steady_state_outlier')]
    ns=np.zeros(n,bool)
    for k in nonsteady:ns|=np.nan_to_num(col(k))>0
    r={'run':stem,'source_volumes':n,'confound_rows':len(rows),'TR_seconds':float(raw.header.get_zooms()[3]),'nonsteady_state_count':int(ns.sum()),'nonsteady_state_indices_0based':np.flatnonzero(ns).tolist(),'mean_FD_mm':float(np.nanmean(fd)),'median_FD_mm':float(np.nanmedian(fd)),'max_FD_mm':float(np.nanmax(fd)),'frames_FD_gt_0_2':int(np.sum(fd>.2)),'frames_FD_gt_0_5':int(np.sum(fd>.5)),'percent_FD_gt_0_5':float(np.mean(fd>.5)*100),'frames_std_DVARS_gt_1_5':int(np.sum(dv>1.5)),'confounds_file':str(conf.relative_to(root)),'outputs':[]}
    r['run']=source_stem
    r['output_prefix']=stem
    r['FD_gt_0_5_indices_0based']=np.flatnonzero(fd>.5).tolist()
    r['std_DVARS_gt_1_5_indices_0based']=np.flatnonzero(dv>1.5).tolist()
    processed=[] if motion_only else sorted(func.glob(stem+'*desc-preproc_bold.nii.gz'))
    for p in processed:
        im=nib.load(p)
        assert im.ndim==4 and im.shape[3]==n
        data=im.get_fdata(dtype=np.float32)
        finite=bool(np.isfinite(data).all())
        assert finite
        r['outputs'].append({'file': str(p.relative_to(root)), 'shape': list(im.shape), 'voxel_spacing': list(map(float, im.header.get_zooms())), 'finite_pixels': finite})
        del data
        gc.collect()
    records.append(r)
    plot_records.append((stem,rows,col))
    time=np.arange(n)*r['TR_seconds']
    fig,axes=plt.subplots(4,1,figsize=(12,8),sharex=True,constrained_layout=True)
    for k in ['trans_x','trans_y','trans_z']:axes[0].plot(time,col(k),label=k)
    axes[0].set_ylabel('Translation (mm)')
    axes[0].legend(ncol=3,loc='upper right',fontsize=8)
    for k in ['rot_x','rot_y','rot_z']:axes[1].plot(time,np.degrees(col(k)),label=k)
    axes[1].set_ylabel('Rotation (deg)')
    axes[1].legend(ncol=3,loc='upper right',fontsize=8)
    axes[2].plot(time,fd,color='#b04c25')
    axes[2].axhline(.5,color='grey',ls='--',lw=.8,label='0.5 mm reference')
    axes[2].set_ylabel('FD (mm)')
    axes[2].legend(fontsize=8)
    axes[3].plot(time,dv,color='#276b8e')
    axes[3].axhline(1.5,color='grey',ls='--',lw=.8)
    axes[3].set_ylabel('Standardized DVARS')
    axes[3].set_xlabel('Time from first saved volume (s)')
    for ax in axes:
     ax.grid(alpha=.2)
     for i in np.flatnonzero(ns):ax.axvspan(time[i],time[i]+r['TR_seconds'],color='orange',alpha=.15)
    fig.suptitle(stem+'\nDashed thresholds are descriptive references; task timing is unverified.',fontsize=11)
    fig.savefig(qc/(stem+'_motion_qc.png'),dpi=150)
    plt.close(fig)
if motion_only:
 (qc/'motion_preview_summary.json').write_text(json.dumps({'status':'MOTION_PREVIEW_ONLY','runs':records},ensure_ascii=False,indent=2),encoding='utf-8')
 print('Motion previews ready for three runs; final preprocessing verification is still pending.')
 raise SystemExit(0)
t1=nib.load(anat/f'{CASE.subject}_desc-preproc_T1w.nii.gz')
t1=nib.as_closest_canonical(t1)
t=t1.get_fdata(dtype=np.float32)
tmask=nib.load(anat/f'{CASE.subject}_desc-brain_mask.nii.gz')
tm=resample_from_to(tmask,t1,order=0).get_fdata()>.5
coords=np.argwhere(tm)
centre=np.median(coords,axis=0).astype(int)
fig,axes=plt.subplots(4,4,figsize=(14,12),constrained_layout=True)
limits=np.percentile(t[tm],[1,99])
zs=np.quantile(coords[:,2],[.25,.45,.65,.82]).astype(int)
for row in range(4):
 if row==0:mask=tm;label='T1 brain extraction'
 else:
  stem=records[row-1]['output_prefix'];p=next(func.glob(stem+'*space-T1w*desc-brain_mask.nii.gz'))
  mask=resample_from_to(nib.load(p),t1,order=0).get_fdata()>.5;label=stem.replace(f'{CASE.subject}_','')
 for j,z in enumerate(zs):
  ax=axes[row,j];ax.imshow(t[:,:,z].T,origin='lower',cmap='gray',vmin=limits[0],vmax=limits[1])
  if mask[:,:,z].any():ax.contour(mask[:,:,z].T,levels=[.5],colors=['#ef5b45'],linewidths=.7)
  ax.axis('off');ax.set_title(f'{label}\nT1 voxel z={z}',fontsize=8)
fig.suptitle('Brain masks in individual T1 space (neurological display)\nContour overlap is a visual check; it does not prove fine-scale registration accuracy.',fontsize=12)
fig.savefig(qc/'brain_masks_in_T1_space.png',dpi=160)
plt.close(fig)
reports=list(out.glob(f'{CASE.subject}.html'))
summary={'status':'OUTPUTS_VERIFIED_VISUAL_REVIEW_PENDING','scope':'Output completeness and descriptive QC; not clinical suitability or activation inference.','runs':records,'subject_report':str(reports[0].relative_to(root)),'threshold_note':'FD 0.2/0.5 mm and standardized DVARS 1.5 counts are descriptive reference thresholds, not automatic exclusion criteria.','not_performed':['slice timing correction (metadata unavailable)','susceptibility distortion correction (suitable inputs unavailable)','FreeSurfer cortical surface reconstruction','spatial smoothing','final nuisance regression','task GLM']}
(qc/'qc_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
cols=['run','source_volumes','confound_rows','TR_seconds','nonsteady_state_count','mean_FD_mm','median_FD_mm','max_FD_mm','frames_FD_gt_0_2','frames_FD_gt_0_5','percent_FD_gt_0_5','frames_std_DVARS_gt_1_5']
with (qc/'motion_summary.tsv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore',delimiter='\t');w.writeheader();w.writerows(records)
print(json.dumps({k:summary[k] for k in ['status','subject_report']},indent=2))
for r in records:print(r['run'],'N',r['source_volumes'],'mean FD',round(r['mean_FD_mm'],4),'max FD',round(r['max_FD_mm'],4),'FD>.5',r['frames_FD_gt_0_5'])
