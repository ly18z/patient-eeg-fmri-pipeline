"""Reproducible, offline subject-level task GLM after fMRIPrep.

Run with the already archived nipreps/fmriprep:25.2.7 image (Python entrypoint).
Raw/BIDS/fMRIPrep files are read only by this script. Timing is user-approved.
"""
from case_config import CASE
import os, json, time, importlib.metadata as md, warnings, gc
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats, ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm import compute_fixed_effects, threshold_stats_img
from nilearn import plotting
ROOT = CASE.root
OUT = ROOT/'derivatives/glm'
QC = ROOT/'qc/glm'
LOG = ROOT/'logs'
for p in [OUT, QC, LOG]: p.mkdir(parents=True, exist_ok=True)
FUNC = ROOT/f'derivatives/fmriprep/{CASE.subject}/func'
ANAT = ROOT/f'derivatives/fmriprep/{CASE.subject}/anat/{CASE.subject}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz'
SPACE = 'space-MNI152NLin2009cAsym_res-2'
START=time.time()
records=[]
CONFIG = dict(task=CASE.task, analysis='single-subject task GLM',
    task_interpretation=CASE.task_settings['description'],
    timing_source=CASE.task_settings['timing_source'],
    time_zero='first saved volume', events_per_run={r:CASE.events(r).to_dict('records') for r in CASE.runs},
    runs=CASE.runs, source_series=[CASE.run(r)['series'] for r in CASE.runs], space=SPACE,
    mask='intersection of configured task-run fMRIPrep brain masks', smoothing_FWHM_mm=5.,
    HRF='SPM canonical + temporal derivative', slice_time_ref=0.,
    highpass='fMRIPrep cosine regressors (128 s cutoff); no duplicate drift/filter step',
    motion='24 parameters: 6 rigid-body, derivatives, squares, derivative squares',
    physiological_noise='first 5 retained aCompCor components from combined WM+CSF mask',
    spike_regressors='one per volume with FD > 0.5 mm OR standardized DVARS > 1.5; plus nonsteady-state flags',
    noise_model='AR(1), Nilearn 100 AR coefficient bins', signal_scaling='voxel-wise percent of temporal mean (Nilearn signal_scaling=0)',
    contrast='task canonical HRF coefficient vs implicit rest baseline; derivative weight 0',
    combination='within-subject inverse-variance fixed effects; runs treated as same task',
    primary_inference='combined-run map, two-sided voxelwise BH FDR q<0.05 within analysis mask',
    secondary_inference='per-run FDR maps for run consistency; each map corrected separately',
    cluster_extent_threshold_voxels=0,
    other_outputs='two-sided Bonferroni FWE p<0.05; unthresholded signed z, t, beta, variance and p maps',
    software={x:md.version(x) for x in ['nilearn','nibabel','numpy','scipy','matplotlib','pandas','scikit-learn']},
    container=CASE.data.get('analysis_image','nipreps/fmriprep:25.2.7'), random_seed=20261026)
def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
def status(phase, **kw):
    write_json(LOG/'task_glm_status.json', dict(status=phase, elapsed_seconds=round(time.time()-START,1), **kw))
    print(phase, json.dumps(kw,ensure_ascii=False), flush=True)
def save_img(data, ref, path):
    hdr=ref.header.copy(); hdr.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(np.asarray(data,np.float32),ref.affine,hdr),path)

def contrast_summary(z_img, effect_img, mask_img, folder, label):
    folder.mkdir(exist_ok=True, parents=True)
    mask=mask_img.get_fdata()>0
    z=z_img.get_fdata()
    beta=effect_img.get_fdata()
    assert np.isfinite(z[mask]).all() and np.isfinite(beta[mask]).all()
    thresholds={}
    result={}
    for method,alpha,tag in [('fdr',.05,'fdr05'),('bonferroni',.05,'fwe05')]:
        with warnings.catch_warnings(record=True) as caught:
            thresholded, th=threshold_stats_img(z_img, mask_img=mask_img,alpha=alpha,
                height_control=method,two_sided=True,cluster_threshold=0)
        thresholded.to_filename(folder/f'{label}_desc-{tag}_zmap.nii.gz')
        td=thresholded.get_fdata()
        result[tag]={'z_threshold':float(th) if np.isfinite(th) else None,
            'positive_voxels':int(np.sum(td[mask]>0)), 'negative_voxels':int(np.sum(td[mask]<0)),
            'warnings':[str(w.message) for w in caught]}
        thresholds[tag]=th
    pv=np.minimum(1.,2*stats.norm.sf(np.abs(z[mask])))
    order=np.argsort(pv)
    count=len(pv)
    qsort=np.minimum.accumulate((pv[order]*count/np.arange(1,count+1))[::-1])[::-1]
    q=np.empty(count)
    q[order]=np.minimum(1,qsort)
    qvol=np.ones(z.shape,np.float32)
    qvol[mask]=q
    save_img(qvol,z_img,folder/f'{label}_desc-fdrQ_statmap.nii.gz')
    result['analysis_voxels']=int(mask.sum())
    result['analysis_volume_mm3']=float(mask.sum()*abs(np.linalg.det(mask_img.affine[:3,:3])))
    result['peak_positive_z']=float(np.max(z[mask]))
    result['peak_negative_z']=float(np.min(z[mask]))
    cluster_rows=[]
    labels_all=np.zeros(mask.shape,np.int16)
    cluster_id=0
    for sign, direction in [(1,'task>rest'),(-1,'rest>task')]:
        active=mask & (sign*z>=thresholds['fdr05']) if np.isfinite(thresholds['fdr05']) else np.zeros(mask.shape,bool)
        labs,nlab=ndimage.label(active,structure=ndimage.generate_binary_structure(3,1))
        components=[]
        for j in range(1,nlab+1):
            ijk=np.argwhere(labs==j); peak=ijk[np.argmax(sign*z[tuple(ijk.T)])]
            components.append((len(ijk),j,peak))
        for size,j,peak in sorted(components,reverse=True,key=lambda x:x[0]):
            cluster_id+=1;labels_all[labs==j]=cluster_id*sign
            xyz=nib.affines.apply_affine(z_img.affine,peak)
            cluster_rows.append(dict(cluster_id=cluster_id,direction=direction,voxels=int(size),
                volume_mm3=float(size*abs(np.linalg.det(z_img.affine[:3,:3]))),
                peak_z=float(z[tuple(peak)]),peak_x_mm=float(xyz[0]),peak_y_mm=float(xyz[1]),
                peak_z_mm=float(xyz[2]),effect_at_peak=float(beta[tuple(peak)])))
    cols=['cluster_id','direction','voxels','volume_mm3','peak_z','peak_x_mm','peak_y_mm','peak_z_mm','effect_at_peak']
    pd.DataFrame(cluster_rows,columns=cols).to_csv(folder/f'{label}_desc-fdr05_clusters.tsv',sep='\t',index=False)
    save_img(labels_all,z_img,folder/f'{label}_desc-fdr05_clusters.nii.gz')
    result['clusters']=cluster_rows
    th=thresholds['fdr05']
    display_th=th if np.isfinite(th) else float(np.max(np.abs(z[mask]))+1)
    vmax=max(6.,min(12.,float(np.max(np.abs(z[mask])))))
    display=plotting.plot_stat_map(z_img,bg_img=str(ANAT),display_mode='z',
        cut_coords=[-10,0,10,20,30,40,50,60,70],threshold=display_th,
        cmap='cold_hot',symmetric_cbar=True,vmax=vmax,black_bg=False,
        title=f'{label}: task vs rest | two-sided FDR q<0.05 | z threshold '+(f'{th:.2f}' if np.isfinite(th) else 'no discoveries'))
    display.savefig(str(QC/f'{label}_fdr05_axial.png'),dpi=160)
    display.close()
    display=plotting.plot_stat_map(z_img,bg_img=str(ANAT),display_mode='ortho',cut_coords=(0,-20,50),
        threshold=display_th,cmap='cold_hot',symmetric_cbar=True,vmax=vmax,black_bg=False,
        title=f'{label}: two-sided FDR q<0.05; warm = task>rest, cool = rest>task')
    display.savefig(str(QC/f'{label}_fdr05_ortho.png'),dpi=160)
    display.close()
    write_json(folder/f'{label}_statistics.json',result)
    return result

def prepare_run(run):
    stem=f'{CASE.subject}_task-{CASE.task}_run-{run}'
    folder=OUT/stem
    folder.mkdir(exist_ok=True)
    bold=FUNC/f'{stem}_{SPACE}_desc-preproc_bold.nii.gz'
    conf=FUNC/f'{stem}_desc-confounds_timeseries.tsv'
    meta=conf.with_suffix('.json')
    df=pd.read_csv(conf,sep='\t')
    metadata=json.loads(meta.read_text())
    N,TR=CASE.acquisition(bold)
    assert len(df)==N
    event=CASE.events(run)
    assert event.onset.min()>=0 and (event.onset+event.duration).max()<=N*TR, 'Events extend beyond the recording'
    event.to_csv(folder/f'{stem}_events.tsv',sep='\t',index=False)
    write_json(folder/f'{stem}_events.json',{'TimingSource':CONFIG['timing_source'],'OnsetReference':CONFIG['time_zero'],
        'Units':'seconds','TaskInterpretation':CONFIG['task_interpretation']})
    motion=['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
    motion_cols=motion+[c+'_derivative1' for c in motion]+[c+'_power2' for c in motion]+[c+'_derivative1_power2' for c in motion]
    cc=[c for c in df.columns if c.startswith('a_comp_cor_') and metadata.get(c,{}).get('Mask')=='combined'][:5]
    assert len(cc)==5, 'Expected five combined-mask CompCor components'
    cosine=[c for c in df.columns if c.startswith('cosine')]
    cols=motion_cols+cc+cosine
    nuisance=df[cols].copy()
    nuisance=nuisance.fillna(0.)
    spike=(df['framewise_displacement'].fillna(0)>.5)|(df['std_dvars'].fillna(0)>1.5)
    ns=[c for c in df if c.startswith('non_steady_state_outlier')]
    if ns:spike|=df[ns].fillna(0).sum(axis=1)>0
    spike_indices=np.flatnonzero(spike.to_numpy())
    for c in cols:
        sd=float(nuisance[c].std(ddof=0))
        nuisance[c]=(nuisance[c]-nuisance[c].mean())/sd
    for i in spike_indices:nuisance[f'spike_{i:03d}']=(np.arange(N)==i).astype(float)
    nuisance.to_csv(folder/f'{stem}_model-confounds.tsv',sep='\t',index=False)
    design=make_first_level_design_matrix(np.arange(N)*TR,events=event,hrf_model='spm + derivative',
        drift_model=None,add_regs=nuisance.to_numpy(),add_reg_names=list(nuisance.columns))
    X=design.to_numpy()
    s=np.linalg.svd(X,compute_uv=False)
    rank=np.linalg.matrix_rank(X)
    assert rank==X.shape[1],(stem,'rank deficient',rank,X.shape)
    condition=float(s[0]/s[-1])
    assert condition<1e8,condition
    task=design[CASE.condition].to_numpy()
    others=design.drop(columns=[CASE.condition]).to_numpy()
    fit=others@np.linalg.lstsq(others,task,rcond=None)[0]
    vif=float(np.sum((task-task.mean())**2)/np.sum((task-fit)**2))
    cor=nuisance.corrwith(pd.Series(task,index=nuisance.index)).sort_values(key=lambda x:abs(x),ascending=False)
    design.to_csv(folder/f'{stem}_design-matrix.tsv',sep='\t',index_label='time_seconds')
    with plt.rc_context({'font.size':8}):
        ax=plotting.plot_design_matrix(design);ax.figure.set_size_inches(15,7)
        ax.set_title(f'Run {run}: rank {rank}/{X.shape[1]}, nominal residual DOF {N-rank}, task VIF {vif:.2f}')
        ax.figure.savefig(QC/f'run-{run}_design_matrix.png',dpi=150,bbox_inches='tight');plt.close(ax.figure)
        corr=design.drop(columns=['constant']).corr().to_numpy()
        fig,ax=plt.subplots(figsize=(11,10));im=ax.imshow(corr,cmap='RdBu_r',vmin=-1,vmax=1)
        labs=list(design.drop(columns=['constant']).columns);ax.set_xticks(range(len(labs)),labs,rotation=90,fontsize=6)
        ax.set_yticks(range(len(labs)),labs,fontsize=6);fig.colorbar(im,ax=ax,shrink=.6)
        ax.set_title(f'Run {run}: design correlation');fig.tight_layout()
        fig.savefig(QC/f'run-{run}_design_correlation.png',dpi=140);plt.close(fig)
        fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True,constrained_layout=True)
        t=np.arange(N)*TR;axes[0].plot(t,task,label='canonical task predictor');axes[0].plot(t,design[CASE.condition+'_derivative'],label='HRF temporal derivative')
        axes[0].legend(fontsize=8);axes[0].set_ylabel('Model predictor')
        axes[1].plot(t,df['framewise_displacement']);axes[1].axhline(.5,ls='--',c='gray');axes[1].set_ylabel('FD (mm)')
        axes[2].plot(t,df['std_dvars']);axes[2].axhline(1.5,ls='--',c='gray');axes[2].set_ylabel('Standardized DVARS');axes[2].set_xlabel('Seconds from first saved volume')
        for ax in axes:
            for row in event.itertuples():ax.axvspan(row.onset,row.onset+row.duration,color='#ffb347',alpha=.16)
            for j in spike_indices:ax.axvline(j*TR,color='#d9534f',alpha=.2,lw=.6)
            ax.grid(alpha=.15)
        fig.suptitle(f'Run {run}: approved task timing and nuisance spikes (red lines)');fig.savefig(QC/f'run-{run}_timing_motion.png',dpi=150);plt.close(fig)
    diag=dict(run=run,source_series=CASE.run(run)['series'],n_volumes=N,TR_seconds=TR,
        regressors=X.shape[1],rank=int(rank),nominal_residual_dof=int(N-rank),condition_number=condition,
        task_VIF=vif,selected_compcor=cc,cosine_regressors=cosine,
        spike_indices_zero_based=spike_indices.tolist(),spike_count=int(len(spike_indices)),
        strongest_task_nuisance_correlations={k:float(v) for k,v in cor.iloc[:8].items()},
        mean_FD_mm=float(df.framewise_displacement.mean()),max_FD_mm=float(df.framewise_displacement.max()))
    write_json(folder/f'{stem}_model-diagnostics.json',diag)
    return bold, design, diag, folder, stem

def main():
    if (OUT/'glm_summary.json').exists():raise RuntimeError('Existing completed analysis: preserve it; use a new output version for reruns')
    status('PREPARING')
    write_json(OUT/'analysis_configuration.json',CONFIG)
    write_json(OUT/'dataset_description.json',{'Name':CASE.root.name+' task GLM','BIDSVersion':'1.11.1','DatasetType':'derivative',
        'GeneratedBy':[{'Name':'Nilearn','Version':md.version('nilearn'),'Description':'fMRI/03_glm/run_task_glm.py'}],
        'SourceDatasets':[{'URL':'../fmriprep','Version':'25.2.7'}]})
    masks=[]
    refs=[]
    for run in CONFIG['runs']:
        mp=FUNC/f'{CASE.subject}_task-{CASE.task}_run-{run}_{SPACE}_desc-brain_mask.nii.gz'
        ref=nib.load(mp)
        refs.append(ref)
        masks.append(ref.get_fdata()>.5)
    assert all(r.shape==refs[0].shape and np.allclose(r.affine,refs[0].affine) for r in refs)
    common=np.logical_and.reduce(masks)
    mask_img=nib.Nifti1Image(common.astype(np.uint8),refs[0].affine)
    mask_file=OUT/f'{CASE.subject}_task-{CASE.task}_{SPACE}_desc-common_mask.nii.gz'
    mask_img.to_filename(mask_file)
    coverage={'per_run_voxels':[int(m.sum()) for m in masks],'common_voxels':int(common.sum()),
        'common_fraction_of_each_run':[float(common.sum()/m.sum()) for m in masks],
        'voxel_volume_mm3':float(abs(np.linalg.det(refs[0].affine[:3,:3]))),'source_masks':'unmodified fMRIPrep run-specific masks'}
    write_json(OUT/'coverage_summary.json',coverage)
    disp=plotting.plot_roi(mask_img,bg_img=str(ANAT),display_mode='z',cut_coords=[-10,0,10,20,30,40,50,60,70],
        title='Common task-run analysis coverage (MNI space)',alpha=.4)
    disp.savefig(str(QC/'common_mask_coverage.png'),dpi=150)
    disp.close()
    effect_imgs=[]
    var_imgs=[]
    dofs=[]
    for run in CONFIG['runs']:
        bold,design,diag,folder,stem=prepare_run(run)
        N,TR=CASE.acquisition(bold)
        raw=nib.load(bold)
        assert raw.shape[:3]==mask_img.shape and np.allclose(raw.affine,mask_img.affine)
        status('FITTING_RUN',run=run,design_regressors=diag['regressors'],spikes=diag['spike_count'])
        model=FirstLevelModel(t_r=TR,slice_time_ref=0.,hrf_model='spm + derivative',drift_model=None,
            mask_img=mask_img,smoothing_fwhm=5.,noise_model='ar1',standardize=False,
            signal_scaling=0,n_jobs=2,minimize_memory=False,random_state=20261026,verbose=1)
        model.fit(str(bold),design_matrices=design)
        contrast=np.array([1. if c==CASE.condition else 0. for c in design.columns])
        maps=model.compute_contrast(contrast,stat_type='t',output_type='all')
        suffix={'effect_size':'effect','effect_variance':'variance','stat':'tmap','z_score':'zmap','p_value':'oneSidedP'}
        for key,value in maps.items():value.to_filename(folder/f'{stem}_contrast-taskVsRest_desc-{suffix[key]}.nii.gz')
        # Full model quality summaries; no voxel time series discarded.
        r2=model.r_square[0];r2.to_filename(folder/f'{stem}_desc-modelR2_statmap.nii.gz')
        r=r2.get_fdata()[common];diag['model_R2_percentiles']=dict(zip(['p05','p50','p95'],map(float,np.percentile(r,[5,50,95]))))
        rho=np.asarray(model.labels_[0],dtype=float)
        diag['AR1_percentiles']=dict(zip(['p05','p50','p95'],map(float,np.percentile(rho,[5,50,95]))))
        resid_img=model.residuals[0];resid=resid_img.get_fdata(dtype=np.float32)[common,:]
        r0=resid[:,:-1]-resid[:,:-1].mean(axis=1,keepdims=True);r1=resid[:,1:]-resid[:,1:].mean(axis=1,keepdims=True)
        lag=(r0*r1).sum(axis=1)/np.sqrt((r0*r0).sum(axis=1)*(r1*r1).sum(axis=1))
        diag['unwhitened_residual_lag1_median']=float(np.nanmedian(lag))
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        axes[0].hist(r[np.isfinite(r)],bins=70,color='#287c8e');axes[0].set_xlabel('Full-model R squared');axes[0].set_ylabel('Voxels')
        axes[1].hist(rho,bins=60,color='#c67931');axes[1].set_xlabel('Estimated AR(1) coefficient')
        fig.suptitle(f'Run {run}: model diagnostics (R squared includes nuisance regressors)')
        fig.savefig(QC/f'run-{run}_fit_diagnostics.png',dpi=150);plt.close(fig)
        del resid,resid_img,r0,r1
        summary=contrast_summary(maps['z_score'],maps['effect_size'],mask_img,folder,stem)
        write_json(folder/f'{stem}_model-diagnostics.json',diag)
        records.append(dict(model=diag,statistics=summary))
        effect_imgs.append(maps['effect_size']);var_imgs.append(maps['effect_variance']);dofs.append(diag['nominal_residual_dof'])
        del model,maps;gc.collect()
        status('RUN_COMPLETE',run=run,positive_FDR_voxels=summary['fdr05']['positive_voxels'],negative_FDR_voxels=summary['fdr05']['negative_voxels'])
    status('COMBINING_RUNS')
    combined=OUT/f'{CASE.subject}_task-{CASE.task}_combined'
    combined.mkdir(exist_ok=True)
    label=f'{CASE.subject}_task-{CASE.task}_combined'
    eff,var,tstat,zstat=compute_fixed_effects(effect_imgs,var_imgs,mask=mask_img,precision_weighted=True,dofs=dofs,return_z_score=True)
    for tag,im in [('effect',eff),('variance',var),('tmap',tstat),('zmap',zstat)]:
        im.to_filename(combined/f'{label}_contrast-taskVsRest_desc-{tag}.nii.gz')
    combined_stats=contrast_summary(zstat,eff,mask_img,combined,label)
    zz=[nib.load(OUT/f'{CASE.subject}_task-{CASE.task}_run-{run}'/f'{CASE.subject}_task-{CASE.task}_run-{run}_contrast-taskVsRest_desc-zmap.nii.gz').get_fdata() for run in CONFIG['runs']]
    ee=[x.get_fdata()[common] for x in effect_imgs]
    from itertools import combinations
    pairs=[]
    for i,j in combinations(range(len(ee)),2):
        a=records[i]['statistics']['fdr05']['z_threshold']
        b=records[j]['statistics']['fdr05']['z_threshold']
        pairs.append(dict(runs=[CASE.runs[i],CASE.runs[j]],
            effect_spatial_Pearson_r=float(np.corrcoef(ee[i],ee[j])[0,1]),
            effect_sign_agreement_fraction=float(np.mean(np.sign(ee[i])==np.sign(ee[j]))),
            positive_FDR_overlap_voxels=int(np.sum(common&(zz[i]>=(a if a is not None else np.inf))&(zz[j]>=(b if b is not None else np.inf))))))
    consistency=(pairs[0].copy() if len(pairs)==1 else {})|{'pairs':pairs}
    write_json(OUT/'run_consistency.json',consistency)
    if len(ee)>1:
        rng=np.random.default_rng(20261026)
        ix=rng.choice(len(ee[0]),min(20000,len(ee[0])),replace=False)
        fig,ax=plt.subplots(figsize=(6,6),constrained_layout=True)
        ax.hexbin(ee[0][ix],ee[1][ix],gridsize=65,bins='log',cmap='viridis')
        ax.set(xlabel=f'Run {CASE.runs[0]} task coefficient',ylabel=f'Run {CASE.runs[1]} task coefficient',title='First two runs: descriptive consistency')
        fig.savefig(QC/'run_consistency.png',dpi=160);plt.close(fig)
    summary = dict(status='COMPLETED_NUMERICALLY_VISUAL_REVIEW_PENDING', configuration=CONFIG, coverage=coverage, runs=records, combined=combined_stats, consistency=consistency, validation={'designs_full_rank': True, 'input_volumes_preserved': True}, elapsed_seconds=round(time.time() - START, 1))
    write_json(OUT/'glm_summary.json',summary)
    status('COMPLETED_NUMERICALLY_VISUAL_REVIEW_PENDING',summary='derivatives/glm/glm_summary.json')
if __name__ == '__main__':
    main()
