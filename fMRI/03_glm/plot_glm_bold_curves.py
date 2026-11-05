"""Recover original voxel-wise fits for display; do not alter inferential maps."""
from case_config import CASE
import os, json, html, gc
os.environ['MPLBACKEND']='Agg'
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np,pandas as pd,nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilearn.maskers import NiftiMasker
from nilearn.glm.first_level.first_level import mean_scaling,run_glm
ROOT = CASE.root
GLM=ROOT/'derivatives/glm'
OUT=ROOT/'derivatives/glm_curves'
QC=ROOT/'qc/glm_curves'
for p in [OUT,QC]:p.mkdir(exist_ok=True,parents=True)
CASE.configure_plotting()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
config=json.loads((GLM/'analysis_configuration.json').read_text())
maskp=GLM/f'{CASE.subject}_task-{CASE.task}_space-MNI152NLin2009cAsym_res-2_desc-common_mask.nii.gz'
maskimg=nib.load(maskp)
mask=maskimg.get_fdata()>.5
zp=GLM/f'{CASE.subject}_task-{CASE.task}_combined/{CASE.subject}_task-{CASE.task}_combined_contrast-taskVsRest_desc-zmap.nii.gz'
zi=nib.load(zp)
z=zi.get_fdata()
assert z.shape==mask.shape and np.allclose(zi.affine,maskimg.affine)
coords=np.column_stack(np.where(mask))
world=nib.affines.apply_affine(maskimg.affine,coords)
regions=[]
for name, cn, sign in [('positive', '正向峰附近', 1), ('negative', '负向峰附近', -1)]:
    loc=int(np.argmax(sign*z[mask]))
    ijk=coords[loc]
    center=world[loc]
    member=np.linalg.norm(world-center,axis=1)<=6.
    rm=np.zeros(mask.shape,dtype=np.uint8)
    rm[tuple(coords[member].T)]=1
    nib.Nifti1Image(rm,maskimg.affine).to_filename(OUT/f'roi-{name}_radius-6mm_mask.nii.gz')
    regions.append(dict(key=name,label=cn,center_MNI_mm=center.tolist(),peak_combined_Z=float(z[tuple(ijk)]),
                        radius_mm=6.,voxel_count=int(member.sum()),member=member))
select=np.logical_or.reduce([r['member'] for r in regions])
selected_coords=coords[select]
for r in regions:r['subset']=r['member'][select]
metadata=dict(status='IN_PROGRESS',selection='6 mm spheres centered on positive/negative extrema of existing combined Z map; intersect common analysis mask',
              interpretation='Same-data illustrative fits, not independent validation; no ROI significance testing or HRF delay search',
              processing='Identical common-mask NiftiMasker 5 mm smoothing, voxel-wise mean scaling, original saved design, AR1 100 bins; voxel fits averaged within each ROI',
              adjustment='Observed adjusted = observed percent-mean signal minus fitted nuisance including intercept; task fit = canonical + temporal derivative',
              config=config,regions=[{k:v for k,v in r.items() if k not in ['member','subset']} for r in regions],runs=[])
allcurves={}
for run in CASE.runs:
    stem=f'{CASE.subject}_task-{CASE.task}_run-{run}'
    folder=GLM/stem
    bold=ROOT/f'derivatives/fmriprep/{CASE.subject}/func/{stem}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
    design=pd.read_csv(folder/f'{stem}_design-matrix.tsv',sep='\t',index_col=0)
    X=design.to_numpy()
    N,TR=CASE.acquisition(bold)
    t=np.arange(N)*TR
    assert len(design)==N and np.allclose(design.index,t)
    print('EXTRACTING',run,flush=True)
    masker=NiftiMasker(mask_img=str(maskp),smoothing_fwhm=5.,standardize=False,detrend=False,verbose=0)
    full=masker.fit_transform(str(bold))
    assert full.shape==(N,int(mask.sum()))
    Y,_=mean_scaling(full[:,select].copy(),axis=0)
    del full
    gc.collect()
    labels,results=run_glm(Y,X,noise_model='ar1',bins=100,n_jobs=2,verbose=0,random_state=20261026)
    theta=np.zeros((X.shape[1],Y.shape[1]))
    for lab,result in results.items():
        use=labels==lab;theta[:,use]=result.theta
    fitted=X@theta
    ci=design.columns.get_loc(CASE.condition)
    di=design.columns.get_loc(CASE.condition+'_derivative')
    canonical=X[:,[ci]]@theta[[ci],:]
    derivative=X[:,[di]]@theta[[di],:]
    task=canonical+derivative
    nuisance=fitted-task
    adjusted=Y-nuisance
    runrec = dict(run=run, source_series=CASE.run(run)['series'], n_volumes=N, regions=[])
    for r in regions:
        member=r['subset'];curve=dict(time_seconds=t,observed_preproc_psc=Y[:,member].mean(axis=1),full_model_fit=fitted[:,member].mean(axis=1),
            fitted_nuisance=nuisance[:,member].mean(axis=1),observed_adjusted=adjusted[:,member].mean(axis=1),
            canonical_task_fit=canonical[:,member].mean(axis=1),temporal_derivative_fit=derivative[:,member].mean(axis=1),task_fit=task[:,member].mean(axis=1),
            unit_canonical_predictor=X[:,ci],unit_derivative_predictor=X[:,di])
        table=pd.DataFrame(curve);table.to_csv(OUT/f'{stem}_roi-{r["key"]}_curves.tsv',sep='\t',index=False)
        allcurves[(run,r['key'])]=curve
        runrec['regions'].append(dict(key=r['key'],mean_canonical_beta=float(theta[ci,member].mean()),mean_derivative_beta=float(theta[di,member].mean()),
                                     adjusted_residual_RMSE=float(np.sqrt(np.mean((curve['observed_adjusted']-curve['task_fit'])**2)))))
    metadata['runs'].append(runrec)
    print('CURVES_READY',run,flush=True)
    del Y,theta,fitted,canonical,derivative,task,nuisance,adjusted,results,masker
    gc.collect()
def shades(ax,run):
    for event in CASE.events(run).itertuples():ax.axvspan(event.onset,event.onset+event.duration,color='#d8e8ee',alpha=.65,zorder=0)
    ax.axhline(0,color='#a8b6c0',linewidth=.6);ax.set_xlim(0,max(v['time_seconds'][-1] for v in allcurves.values()));ax.grid(axis='y',alpha=.15);ax.set_xlabel('时间（s）')
    ax.set_ylabel('BOLD变化（时间均值的%）')
fig,axes=plt.subplots(2,len(CASE.runs),squeeze=False,figsize=(15,7),sharex=True,layout='constrained')
for row,r in enumerate(regions):
    for col,run in enumerate(CASE.runs):
        ax=axes[row,col];v=allcurves[(run,r['key'])];t=v['time_seconds'];shades(ax,run)
        ax.plot(t,v['observed_adjusted'],color='#267ea2',lw=1.15,label='混杂调整后BOLD')
        ax.plot(t,v['canonical_task_fit'],color='#787f87',lw=1.4,ls='--',label='标准HRF任务项')
        ax.plot(t,v['task_fit'],color='#dc702f',lw=2,label='任务拟合（含时间导数）')
        ax.set_title(f'Run {int(run)} · {r["label"]}（{r["voxel_count"]}体素）',loc='left',fontweight='bold')
        values=np.r_[v['observed_adjusted'],v['canonical_task_fit'],v['task_fit']];lo,hi=values.min(),values.max();pad=(hi-lo)*.13
        ax.set_ylim(lo-pad,hi+pad)
for row,r in enumerate(regions):
    values=np.concatenate([allcurves[(run,r['key'])][key] for run in CASE.runs for key in ['observed_adjusted','canonical_task_fit','task_fit']])
    lo,hi=values.min(),values.max();pad=(hi-lo)*.13
    for ax in axes[row]:ax.set_ylim(lo-pad,hi+pad)
axes[0,0].legend(loc='upper left',fontsize=8,ncols=1,framealpha=.9)
fig.savefig(QC/'task_prediction_vs_bold.png',dpi=165)
fig.savefig(QC/'task_prediction_vs_bold.svg')
plt.close(fig)
fig,axes=plt.subplots(2,len(CASE.runs),squeeze=False,figsize=(15,7),sharex=True,layout='constrained')
for row,r in enumerate(regions):
    for col,run in enumerate(CASE.runs):
        ax=axes[row,col];v=allcurves[(run,r['key'])];t=v['time_seconds'];shades(ax,run)
        ax.plot(t,v['observed_preproc_psc'],color='#267ea2',lw=1.1,label='预处理后BOLD（未回归混杂）')
        ax.plot(t,v['full_model_fit'],color='#dc702f',lw=1.7,label='全模型拟合（含混杂项）')
        ax.set_title(f'Run {int(run)} · {r["label"]}',loc='left',fontweight='bold')
for row,r in enumerate(regions):
    values=np.concatenate([allcurves[(run,r['key'])][key] for run in CASE.runs for key in ['observed_preproc_psc','full_model_fit']])
    lo,hi=values.min(),values.max();pad=(hi-lo)*.13
    for ax in axes[row]:ax.set_ylim(lo-pad,hi+pad)
axes[0,0].legend(loc='upper left',fontsize=8,framealpha=.9)
fig.savefig(QC/'preproc_bold_full_model.png',dpi=165)
plt.close(fig)
anat=nib.load(ROOT/f'derivatives/fmriprep/{CASE.subject}/anat/{CASE.subject}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz')
from nilearn import plotting
for r in regions:
    disp=plotting.plot_roi(str(OUT/f'roi-{r["key"]}_radius-6mm_mask.nii.gz'),bg_img=str(anat.get_filename()),
        cut_coords=r['center_MNI_mm'],display_mode='ortho',alpha=.6,cmap='autumn',colorbar=False,title=r['key']+' peak: 6 mm sphere / common mask')
    disp.savefig(QC/f'roi-{r["key"]}_location.png',dpi=120);disp.close()
metadata['status']='CURVES_READY'
metadata['figure_files']=['qc/glm_curves/task_prediction_vs_bold.png','qc/glm_curves/preproc_bold_full_model.png','qc/glm_curves/roi-positive_location.png','qc/glm_curves/roi-negative_location.png']
(OUT/'curve_summary.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
rows=''.join('<tr><td>'+html.escape(r['label'])+'</td><td>'+', '.join(f'{x:.1f}' for x in r['center_MNI_mm'])+'</td><td>'+str(r['voxel_count'])+'</td></tr>' for r in regions)
section='''<section id="bold-curves"><h2>任务预测与实际BOLD</h2><p>蓝线：去除已拟合混杂项后的区域平均BOLD；橙线：标准HRF任务项＋时间导数；灰虚线：标准HRF任务项。阴影表示该run配置的任务区块。纵轴为原时间均值的百分比尺度，未对曲线逐一归一化。</p><figure><img src="qc/glm_curves/task_prediction_vs_bold.png" alt="各run正负峰附近的实际BOLD与任务拟合曲线"><figcaption>各run分别拟合；区域内各体素沿用原模型，之后取平均。混杂控制包含头动、aCompCor、漂移、异常帧和截距。</figcaption></figure><p>两处区域均取自本次合并Z图的正／负峰，半径6 mm并限定在共同掩膜内。它们用于展示拟合，不是预定义运动区，也不是独立验证。橙线幅度来自同一批数据的拟合。</p><div class="table"><table><thead><tr><th>区域</th><th>中心MNI坐标（mm）</th><th>体素数</th></tr></thead><tbody>'''+rows+'''</tbody></table></div><details><summary>预处理后BOLD与全模型拟合</summary><p>这里保留混杂相关变化；橙线包含任务和全部混杂项，贴合程度不能作为任务效应强度。</p><figure><img src="qc/glm_curves/preproc_bold_full_model.png" alt="预处理后BOLD与全模型拟合"></figure></details><details><summary>区域位置与曲线数据</summary><figure><img src="qc/glm_curves/roi-positive_location.png" alt="正向峰球形区域位置"></figure><figure><img src="qc/glm_curves/roi-negative_location.png" alt="负向峰球形区域位置"></figure><p>'''
links=[]
for run in CASE.runs:
    for r in regions:
        name=f'{CASE.subject}_task-{CASE.task}_run-{run}_roi-{r["key"]}_curves.tsv'
        links.append(f'<a href="derivatives/glm_curves/{name}">Run {int(run)} · {r["label"]}曲线</a>')
section+=' · '.join(links)+'''</p><p><a href="derivatives/glm_curves/curve_summary.json">提取方法</a></p></details></section>'''
(OUT/'report_section.html').write_text(section,encoding='utf-8')
print('CURVES_COMPLETE',json.dumps(metadata['regions'],ensure_ascii=False),flush=True)
