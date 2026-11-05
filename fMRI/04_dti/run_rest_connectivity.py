from case_config import CASE
import json,time,gc
import numpy as np,pandas as pd,nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilearn.maskers import NiftiMasker
from nilearn import plotting

R = CASE.root;O=R/'analysis_rest_dti';OUT=O/'rest';Q=O/'qc';OUT.mkdir(parents=True,exist_ok=True);Q.mkdir(exist_ok=True)
CASE.configure_plotting();plt.rcParams.update({'axes.unicode_minus':False,'font.size':10})
start=time.time()
def status(s,**kw):
    (OUT/'status.json').write_text(json.dumps({'status':s,'elapsed_seconds':time.time()-start,**kw},indent=2),encoding='utf-8');print(s,flush=True)
status('EXTRACTING')
F=R/f'derivatives/fmriprep/{CASE.subject}/func';prefix=f'{CASE.subject}_task-rest';space='space-MNI152NLin2009cAsym_res-2'
bold=F/f'{prefix}_{space}_desc-preproc_bold.nii.gz';mask=F/f'{prefix}_{space}_desc-brain_mask.nii.gz'
df=pd.read_csv(F/f'{prefix}_desc-confounds_timeseries.tsv',sep='\t');meta=json.loads((F/f'{prefix}_desc-confounds_timeseries.json').read_text())
info=json.loads((O/'anatomy/motor_ROIs.json').read_text(encoding='utf-8'));labels=[info['labels'][str(i)] for i in range(1,7)];names=[x['short'] for x in labels]
masker=NiftiMasker(mask_img=str(mask),standardize=False,detrend=False,smoothing_fwhm=None)
raw=masker.fit_transform(str(bold));assert raw.shape[0]==len(df) and np.isfinite(raw).all()
mean=raw.mean(axis=0);intensity_floor=float(.01*np.median(mean[mean>0]));good=(mean>intensity_floor)&(raw.std(axis=0)>1e-8)
initial_mask=nib.load(mask);effective=initial_mask.get_fdata()>.5;initial_count=int(effective.sum());effective[effective]=good
effective_path=OUT/'effective_signal_mask.nii.gz';nib.save(nib.Nifti1Image(effective.astype(np.uint8),initial_mask.affine),effective_path)
masker=NiftiMasker(mask_img=str(effective_path),standardize=False,detrend=False,smoothing_fwhm=None).fit()
raw=raw[:,good];mean=mean[good];Y=100*(raw/mean-1);del raw;gc.collect()
print('SIGNAL_MASK',int(effective.sum()),'excluded',initial_count-int(effective.sum()),flush=True)
region=nib.load(O/'anatomy/motor_ROIs_space-MNI2009_desc-restCovered_dseg.nii.gz').get_fdata()[effective].astype(int)
assert len(region)==Y.shape[1]
motion=['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z'];cols=motion+[x+'_derivative1' for x in motion]+[x+'_power2' for x in motion]+[x+'_derivative1_power2' for x in motion]
cc=[x for x in df if x.startswith('a_comp_cor_') and meta.get(x,{}).get('Mask')=='combined'][:5];assert len(cc)==5
base=df[cols+cc].fillna(0).to_numpy();base=(base-base.mean(axis=0))/base.std(axis=0)
spike=(df.framewise_displacement.fillna(0).to_numpy()>CASE.data.get('rest',{}).get('fd_mm',.5))|(df.std_dvars.fillna(0).to_numpy()>CASE.data.get('rest',{}).get('std_dvars',1.5))
ns=[x for x in df if x.startswith('non_steady_state_outlier')]
if ns:spike|=df[ns].fillna(0).sum(axis=1).to_numpy()>0
valid=~spike;N,TR=CASE.acquisition(bold);t=np.arange(N)*TR
# Joint regression of nuisance, out-of-band DCT frequencies and spike indicators.
k=np.arange(1,N);freq=k/(2*N*TR);band=CASE.data.get('rest',{}).get('band_hz',[.01,.1]);outside=(freq<band[0])|(freq>band[1])
C=np.cos(np.pi*(np.arange(N)[:,None]+.5)*k[outside][None,:]/N)
S=np.eye(N)[:,spike]
X=np.column_stack([np.ones(N),base,C,S]);u,s,vh=np.linalg.svd(X,full_matrices=False);rank=int(np.sum(s>s[0]*1e-10));u=u[:,:rank]
assert N-rank>=20,(rank,N)
pd.DataFrame(X,columns=['intercept']+cols+cc+[f'filter_DCT_{int(i):03d}' for i in k[outside]]+[f'spike_{i:03d}' for i in np.where(spike)[0]]).to_csv(OUT/'denoising_design.tsv',sep='\t',index=False)
status('DENOISING',rank=rank,spikes=int(spike.sum()))
clean=Y-u@(u.T@Y)
roiRaw=np.column_stack([Y[:,region==i].mean(axis=1) for i in range(1,7)])
roiClean=np.column_stack([clean[:,region==i].mean(axis=1) for i in range(1,7)])
gs=Y.mean(axis=1);gs=gs-u@(u.T@gs)
roiGSR=roiClean-gs[:,None]*((gs@roiClean)/(gs@gs))[None,:]
def corr(x):return np.corrcoef(x[valid].T)
mat=corr(roiClean);gsr=corr(roiGSR)
for name,value in [('correlation',mat),('fisherZ',np.arctanh(np.clip(mat,-.999999,.999999))),('correlation_GSR_sensitivity',gsr)]:
    if name=='fisherZ':np.fill_diagonal(value,np.nan)
    pd.DataFrame(value,index=names,columns=names).to_csv(OUT/f'motor_FC_{name}.tsv',sep='\t')
table=pd.DataFrame({'time_seconds':t,'retained_for_FC':valid.astype(int),'FD_mm':df.framewise_displacement,'std_DVARS':df.std_dvars})
for i,name in enumerate(names):
    table[name+'_raw_psc']=roiRaw[:,i];table[name+'_clean_psc']=np.where(valid,roiClean[:,i],np.nan)
table.to_csv(OUT/'ROI_timeseries.tsv',sep='\t',index=False)
table.loc[valid].to_csv(OUT/'ROI_timeseries_retained.tsv',sep='\t',index=False)
np.savetxt(OUT/'retained_volume_indices_0based.txt',np.flatnonzero(valid),fmt='%d')
fig,axes=plt.subplots(1,2,figsize=(13,5.8),layout='constrained')
for ax,value,title in zip(axes,[mat,gsr],['主要结果：无全局信号回归','敏感性对照：加入全局信号回归']):
    im=ax.imshow(value,cmap='RdBu_r',vmin=-1,vmax=1);ax.set_xticks(range(6),names,rotation=45,ha='right');ax.set_yticks(range(6),names);ax.set_title(title)
    for i in range(6):
        for j in range(6):ax.text(j,i,f'{value[i,j]:.2f}',ha='center',va='center',color='white' if abs(value[i,j])>.55 else 'black',fontsize=9)
fig.colorbar(im,ax=axes,shrink=.7,label='Pearson r');fig.savefig(Q/'rest_motor_FC_matrices.png',dpi=155);plt.close(fig)
fig,axes=plt.subplots(3,2,figsize=(15,8),sharex=True,layout='constrained')
for i,ax in enumerate(axes.flat):
    ax.plot(t,roiRaw[:,i],color='#a4b5c1',lw=.8,label='去噪前');ax.plot(t,np.where(valid,roiClean[:,i],np.nan),color='#137d9b',lw=1,label='去噪后')
    for n in np.flatnonzero(spike):ax.axvline(n*TR,color='#c4614d',alpha=.3,lw=.7)
    ax.set_title(names[i],loc='left');ax.set_ylabel('时间均值的%');ax.grid(alpha=.15)
axes[0,0].legend(fontsize=9);axes[-1,0].set_xlabel('时间（s）');axes[-1,1].set_xlabel('时间（s）');fig.savefig(Q/'rest_ROI_timeseries.png',dpi=145);plt.close(fig)
status('SEED_MAPS')
data=clean[valid];data=data-data.mean(axis=0);norm=np.linalg.norm(data,axis=0);norm[norm<1e-10]=np.inf
bg=R/f'derivatives/fmriprep/{CASE.subject}/anat/{CASE.subject}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz'
seedpaths=[]
for i in [0,1,2,3]:
    seed=roiClean[valid,i];seed=seed-seed.mean();values=(seed@data)/(np.linalg.norm(seed)*norm)
    assert np.isfinite(values).all() and np.max(np.abs(values))<=1+1e-6
    image=masker.inverse_transform(values.astype(np.float32));p=OUT/f'seed-{names[i].replace(" ","")}_desc-correlation.nii.gz';image.to_filename(p)
    z=masker.inverse_transform(np.arctanh(np.clip(values,-.999999,.999999)).astype(np.float32));z.to_filename(OUT/(p.name.replace('correlation','fisherZ')))
    disp=plotting.plot_stat_map(image,bg_img=str(bg),display_mode='z',cut_coords=[20,35,50,65,75],threshold=.2,vmax=.8,cmap='cold_hot',symmetric_cbar=True,title=names[i]+' resting FC | display |r| >= 0.2 (not significance)')
    qp=Q/f'rest_seed_{i+1}.png';disp.savefig(qp,dpi=145);disp.close();seedpaths.append(str(qp.relative_to(O)))
del data;gc.collect()
# Exploratory interhemispheric lag curves; do not time-shift the primary FC data.
lagrows=[];fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
for i,j in [(0,1),(2,3),(4,5)]:
    vals=[]
    for lag in range(-5,6):
        li=np.arange(N);ri=li+lag;ok=(ri>=0)&(ri<N);li=li[ok];ri=ri[ok];ok=valid[li]&valid[ri];li=li[ok];ri=ri[ok]
        value=float(np.corrcoef(roiClean[li,i],roiClean[ri,j])[0,1]);vals.append(value);lagrows.append({'pair':names[i]+' / '+names[j],'right_shift_seconds':lag*TR,'correlation':value,'paired_samples':len(li)})
    ax.plot(np.arange(-5,6)*TR,vals,marker='o',label=names[i][2:])
ax.axvline(0,color='gray',ls='--');ax.set_xlabel('右侧时间序列相对左侧的取样滞后（s）');ax.set_ylabel('Pearson r');ax.legend();ax.grid(alpha=.2);fig.savefig(Q/'rest_interhemispheric_lag.png',dpi=150);plt.close(fig)
pd.DataFrame(lagrows).to_csv(OUT/'interhemispheric_lag_exploratory.tsv',sep='\t',index=False)
halves=[]
for select in [valid&(t<N*TR/2),valid&(t>=N*TR/2)]:halves.append(np.corrcoef(roiClean[select].T).tolist())
pd.DataFrame([{'ROI':x['name'],'coverage_percent':100*x['standard_coverage_fraction'],'native_coverage_percent':100*x['native_BOLD_coverage_fraction'],'valid_voxels':x['standard_valid_voxels']} for x in info['coverage']]).to_csv(OUT/'ROI_coverage.tsv',sep='\t',index=False)
status('SAVING_DENOISED')
clean[spike]=0;masker.inverse_transform(clean.astype(np.float32)).to_filename(OUT/f'{CASE.subject}_task-rest_space-MNI2009_desc-denoised_bold.nii.gz')
result={'status':'COMPLETE_VISUAL_REVIEW_PENDING','frames':N,'TR_seconds':TR,'censored_frames':int(spike.sum()),'censored_indices_0based':np.flatnonzero(spike).tolist(),'retained_frames':int(valid.sum()),'retained_observation_seconds':float(valid.sum()*TR),
 'denoising':{'motion_regressors':24,'combined_aCompCor':cc,'filter':f'Joint DCT out-of-band regression, retained {band[0]}-{band[1]} Hz','out_of_band_DCT_columns':int(outside.sum()),'spike_regressors':int(spike.sum()),'design_columns':X.shape[1],'design_rank':rank,'residual_dimension':N-rank,'spatial_smoothing_mm':0,'global_signal_regression_primary':False,'GSR_sensitivity':True,'sample_handling':f'Spike indicators included during joint regression; censored observations excluded from FC/lag; denoised NIfTI retains {N} volumes with censored frames zero, sample indices supplied'},
 'signal_mask':{'initial_voxels':initial_count,'retained_voxels':int(effective.sum()),'minimum_temporal_mean':intensity_floor,'rule':'finite, nonconstant, mean greater than 1% of positive brain-mask median mean'},'labels':labels,'coverage':[{**x,'effective_signal_voxels':int((region==x['id']).sum()),'effective_signal_coverage_fraction':float((region==x['id']).sum()/max(1,x['standard_roi_voxels']))} for x in info['coverage']],'correlation_matrix':mat.tolist(),'correlation_GSR_sensitivity':gsr.tolist(),'split_half_descriptive_matrices':halves,'seed_figures':seedpaths,'lag_peak_descriptive':[max([x for x in lagrows if x['pair']==pair],key=lambda x:x['correlation']) for pair in sorted(set(x['pair'] for x in lagrows))],
 'limitations':['Single patient; no normative abnormality or causal/structural connectivity inference','ROI definitions are anatomical, not hand-specific or lesion-side confirmed','Lag curves select best of 11 tested lags in same data; not HRF latency or causal evidence; primary FC unchanged','Inspect this case preprocessing log for actual slice timing and susceptibility correction','No voxelwise significance tests; |r|>=0.2 is display threshold only','Split halves share full-run nuisance/filter fit and are descriptive, not independent validation'],'elapsed_seconds':time.time()-start}
(OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
status('COMPLETE',retained_frames=int(valid.sum()),LR_Precentral_r=float(mat[0,1]));print(json.dumps({'retained':int(valid.sum()),'rank':rank,'matrix':mat.tolist(),'lags':result['lag_peak_descriptive']},ensure_ascii=False),flush=True)
