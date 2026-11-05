"""Annotate existing voxel-FDR clusters without recomputing or changing GLM."""
from case_config import CASE
import json, subprocess, time, xml.etree.ElementTree as ET
import numpy as np,pandas as pd,nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilearn import plotting
R = CASE.root
O=R/'analysis_glm_regions'
A=O/'atlas'
Q=O/'qc'
T=R/f'derivatives/fmriprep/{CASE.subject}/anat'
L=R/'derivatives/lesion/anatomy'
for p in [O,A,Q,O/'tables']:p.mkdir(exist_ok=True)
start=time.time()
ref=R/f'derivatives/glm/{CASE.subject}_task-{CASE.task}_combined/{CASE.subject}_task-{CASE.task}_combined_contrast-taskVsRest_desc-zmap.nii.gz'
refim=nib.load(ref)
background=T/f'{CASE.subject}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz'
cn=['额极','岛叶皮层','额上回','额中回','额下回三角部','额下回盖部','中央前回','颞极','颞上回前部','颞上回后部','颞中回前部','颞中回后部','颞中回颞枕部','颞下回前部','颞下回后部','颞下回颞枕部','中央后回','顶上小叶','缘上回前部','缘上回后部','角回','枕外侧皮层上部','枕外侧皮层下部','距状沟内皮层','额内侧皮层','辅助运动区参考（Juxtapositional Lobule）','胼胝体下皮层','旁扣带回','扣带回前部','扣带回后部','楔前叶皮层','楔叶皮层','眶额皮层','海马旁回前部','海马旁回后部','舌回','颞梭状皮层前部','颞梭状皮层后部','颞枕梭状皮层','枕梭状回','额叶岛盖皮层','中央岛盖皮层','顶叶岛盖皮层','极平面','Heschl回（横颞回）','颞平面','距状沟上皮层','枕极']
source=R/'analysis_rest_dti/work/atlas/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz'
xml=source.parent/'HarvardOxford-Cortical.xml'
im=nib.load(source)
v=im.get_fdata().astype(int)
indices=np.indices(im.shape,dtype=np.float32)
wx=sum(im.affine[0,i]*indices[i] for i in range(3))+im.affine[0,3]
del indices
split=np.where(v>0,2*v-(wx<0),0).astype(np.uint8)
splitpath=A/'HO_cortical_hemisphereSplit_space-MNI6_dseg.nii.gz'
nib.save(nib.Nifti1Image(split,im.affine),splitpath)
cortex_labels={0:{'name_cn':'未分配','name_en':'Unassigned','hemisphere':'—'}}
for e in ET.parse(xml).findall('.//label'):
    i=int(e.attrib['index'])
    for side,offset in [('L',1),('R',2)]:
     cortex_labels[2*i+offset]={'name_cn':('左' if side=='L' else '右')+cn[i],'name_en':('Left ' if side=='L' else 'Right ')+e.text.strip(),'hemisphere':side,'source_index':i+1}
sub_source=R/'work/lesion_atlas/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz'
sub_xml=sub_source.parent/'HarvardOxford-Subcortical.xml'
sub_names=json.loads((L/'atlas_labels.json').read_text())
sub_cn={2:'左大脑白质',3:'左大脑皮层',4:'左侧脑室',10:'左丘脑',11:'左尾状核',12:'左壳核',13:'左苍白球',16:'脑干',17:'左海马',18:'左杏仁核',26:'左伏隔核',41:'右大脑白质',42:'右大脑皮层',43:'右侧脑室',49:'右丘脑',50:'右尾状核',51:'右壳核',52:'右苍白球',53:'右海马',54:'右杏仁核',58:'右伏隔核'}
sub_labels={0:{'name_cn':'未分配','name_en':'Unassigned','hemisphere':'—'}}
for k,name in sub_names.items():
 i=int(k);sub_labels[i]={'name_cn':sub_cn[i],'name_en':name,'hemisphere':'L' if name.startswith('Left') else 'R' if name.startswith('Right') else 'midline'}
commands=[]
mapped={}
labels={'cortical':cortex_labels,'subcortical':sub_labels}
for key, src in [('cortical', splitpath), ('subcortical', sub_source)]:
    dest=A/f'HO_{key}_space-MNI152NLin2009cAsym_res-2_dseg.nii.gz'
    cmd=list(map(str,['antsApplyTransforms','-d','3','--float','1','-i',src,'-r',ref,'-o',dest,'-n','NearestNeighbor','-t',T/f'{CASE.subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5','-t',L/'from-MNI6_to-T1w_1Warp.nii.gz','-t',L/'from-MNI6_to-T1w_0GenericAffine.mat']))
    commands.append(cmd)
    with (A/f'{key}_transform.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    out=nib.load(dest)
    assert out.shape==refim.shape and np.allclose(out.affine,refim.affine)
    mapped[key]=np.rint(out.get_fdata()).astype(int)
    for sign,side in [(-1,'L'),(1,'R')]:
     ids=[i for i,x in labels[key].items() if x['hemisphere']==side];ijk=np.argwhere(np.isin(mapped[key],ids));x=nib.affines.apply_affine(out.affine,ijk)[:,0];assert np.mean(sign*x>0)>.95,('Laterality mismatch',key,side)
    disp=plotting.plot_roi(str(dest),bg_img=str(background),display_mode='z',cut_coords=[-10,5,20,35,50,65,80],alpha=.5,colorbar=False,title='Harvard-Oxford '+key+' on GLM MNI2009 grid')
    disp.savefig(Q/f'atlas_{key}_MNI2009.png',dpi=150)
    disp.close()
pd.DataFrame([{'atlas':k,'label_id':i,**x} for k,d in labels.items() for i,x in d.items()]).to_csv(A/'label_dictionary.tsv',sep='\t',index=False)
config = {'status': 'COMPLETE_NUMERICAL_QC_PENDING', 'space': 'MNI152NLin2009cAsym res-2; exact original GLM image grid', 'atlas': 'Harvard-Oxford cortical + subcortical maximum-probability threshold 25%, 1mm source', 'mapping': 'MNI6 -> patient T1 -> GLM MNI2009; composed transforms, one nearest-neighbor resampling; cortical hemisphere split before mapping', 'subcortical_encoding': 'FreeSurfer tutorial copy uses aseg IDs, explicitly checked against original XML ordering', 'threshold': 'Existing two-sided voxelwise BH FDR q<0.05; unchanged', 'cluster_definition': 'Existing 6-neighbor components; no added size threshold; clusters are grouping, not cluster-level corrected inference', 'peak_definition': 'One maximum signed Z per existing component (most negative Z for rest>task); not all local maxima', 'overlap_definition': 'Hard-label voxel counts / entire cluster voxel count x100, including unassigned; each atlas separately sums to100%; not atlas membership probability or activation probability', 'hemisphere': 'Atlas source hemisphere retained; peak hemisphere additionally from MNI x sign', 'commands': commands, 'sources': [{'file': str(p.relative_to(R))} for p in [source, xml, sub_source, sub_xml, T / f'{CASE.subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5', L / 'from-MNI6_to-T1w_1Warp.nii.gz', L / 'from-MNI6_to-T1w_0GenericAffine.mat']], 'limitations': ['Population atlas localization estimates, not individual boundary verification', 'HO does not provide detailed cerebellar parcellation; leave uncovered points unassigned', 'No nearest-label extrapolation, no extra GM masking, no new GLM or multiple-comparison test', 'Voxelwise significance does not imply every reported connected component has cluster-level significance', 'Precentral and SMA reference labels do not confirm individual hand-M1 or functional necessity']}
peakrows=[]
overlaps=[]
totals=[]
verification=[]
global_peaks=[]
for run in ['run-'+r for r in CASE.runs]+['combined']:
    name=f'{CASE.subject}_task-{CASE.task}_'+run
    folder=R/'derivatives/glm'/name
    zim=nib.load(folder/(name+'_contrast-taskVsRest_desc-zmap.nii.gz'))
    z=zim.get_fdata()
    clim=nib.load(folder/(name+'_desc-fdr05_clusters.nii.gz'))
    cl=np.rint(clim.get_fdata()).astype(int)
    table=pd.read_csv(folder/(name+'_desc-fdr05_clusters.tsv'),sep='\t')
    stats=json.loads((folder/(name+'_statistics.json')).read_text())
    assert zim.shape==refim.shape and np.allclose(zim.affine,refim.affine) and np.allclose(clim.affine,zim.affine)
    voxel=abs(np.linalg.det(zim.affine[:3,:3]))
    for row in table.to_dict('records'):
        sign=1 if row['direction']=='task>rest' else -1
        cid=int(row['cluster_id'])
        m=cl==sign*cid
        n=int(m.sum())
        xyz=np.array([row['peak_x_mm'],row['peak_y_mm'],row['peak_z_mm']])
        ijk=np.rint(nib.affines.apply_affine(np.linalg.inv(zim.affine),xyz)).astype(int)
        ix=tuple(ijk)
        p={'run':run,**row,'peak_hemisphere':'L' if xyz[0]<0 else 'R' if xyz[0]>0 else 'midline'}
        for key, a in mapped.items():
            lab=int(a[ix])
            p[key+'_peak_label_id']=lab
            p[key+'_peak_region_cn']=labels[key][lab]['name_cn']
            p[key+'_peak_region_en']=labels[key][lab]['name_en']
            values,counts=np.unique(a[m],return_counts=True)
            entries=[]
            for lab,c in sorted(zip(values.tolist(),counts.tolist()),key=lambda x:x[1],reverse=True):
             entry={'run':run,'cluster_id':cid,'direction':row['direction'],'atlas':key,'label_id':lab,'region_cn':labels[key][lab]['name_cn'],'region_en':labels[key][lab]['name_en'],'voxels':c,'volume_mm3':c*voxel,'percent_of_entire_cluster':100*c/n,'cluster_voxels':n};overlaps.append(entry);entries.append(entry)
            p[key+'_assigned_percent']=100*float(np.count_nonzero(a[m]))/n
            p[key+'_top_overlap']='；'.join(x['region_cn']+f' {x["percent_of_entire_cluster"]:.1f}%' for x in entries[:4])
        peakrows.append(p)
    for sign, direction in [(1, 'task>rest'), (-1, 'rest>task')]:
        m=cl*sign>0
        nn=int(m.sum())
        items=[x for x in peakrows if x['run']==run and x['direction']==direction]
        best=max(items,key=lambda x:abs(x['peak_z'])) if items else None
        if best:global_peaks.append(best)
        verification.append({'run':run,'direction':direction,'significant_voxels':nn,'clusters':len(items),'peak_z':best['peak_z'] if best else None})
        for key,a in mapped.items():
         for lab in np.unique(a[m]):
          reg=m&(a==lab);idx=np.argwhere(reg);peak=idx[np.argmax(sign*z[reg])];xyz=nib.affines.apply_affine(zim.affine,peak)
          totals.append({'run':run,'direction':direction,'atlas':key,'label_id':int(lab),'region_cn':labels[key][int(lab)]['name_cn'],'region_en':labels[key][int(lab)]['name_en'],'significant_voxels':len(idx),'volume_mm3':len(idx)*voxel,'percent_of_direction_voxels':100*len(idx)/nn,'regional_peak_z':float(z[tuple(peak)]),'regional_peak_x_mm':xyz[0],'regional_peak_y_mm':xyz[1],'regional_peak_z_mm':xyz[2]})
    peakdf=pd.DataFrame([x for x in peakrows if x['run']==run]).sort_values('peak_z',key=lambda s:abs(s),ascending=False)
    peakdf.to_csv(O/'tables'/f'{run}_peak_labels.tsv',sep='\t',index=False)
    pd.DataFrame([x for x in overlaps if x['run']==run]).to_csv(O/'tables'/f'{run}_cluster_overlap.tsv',sep='\t',index=False)
    print('ANNOTATED',run,len(table),'clusters',flush=True)
pd.DataFrame(peakrows).to_csv(O/'tables/all_peak_labels.tsv',sep='\t',index=False)
pd.DataFrame(overlaps).to_csv(O/'tables/all_cluster_overlap.tsv',sep='\t',index=False)
pd.DataFrame(totals).to_csv(O/'tables/all_region_totals.tsv',sep='\t',index=False)
pd.DataFrame(global_peaks).to_csv(O/'tables/strongest_peak_per_direction.tsv',sep='\t',index=False)
CASE.configure_plotting()
plt.rcParams['axes.unicode_minus']=False
for row in global_peaks:
 name=f'{CASE.subject}_task-{CASE.task}_'+row['run'];zpath=R/'derivatives/glm'/name/(name+'_desc-fdr05_zmap.nii.gz')
 coords=(row['peak_x_mm'],row['peak_y_mm'],row['peak_z_mm'])
 disp=plotting.plot_stat_map(str(zpath),bg_img=str(background),display_mode='ortho',cut_coords=coords,threshold=.01,vmax=9,symmetric_cbar=True,title=f'{row["run"]} | {row["direction"]} | Z={row["peak_z"]:.2f}')
 disp.add_contours(str(A/'HO_cortical_space-MNI152NLin2009cAsym_res-2_dseg.nii.gz'),levels=[.5],colors=['#e2c347'],linewidths=.4)
 tag='positive' if row['direction']=='task>rest' else 'negative';disp.savefig(Q/f'peak_{row["run"]}_{tag}.png',dpi=145);disp.close()
config['elapsed_seconds']=time.time()-start
(O/'configuration.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
summary = {'status': 'COMPLETE_INITIAL_VISUAL_REVIEW_PENDING', 'maps': verification, 'strongest_peaks': global_peaks, 'cluster_count': len(peakrows), 'overlap_rows': len(overlaps), 'configuration': 'configuration.json'}
(O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False),flush=True)
