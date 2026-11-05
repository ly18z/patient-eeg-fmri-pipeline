from case_config import CASE
import json,shutil,xml.etree.ElementTree as ET
import numpy as np,pandas as pd,nibabel as nib
R = CASE.root
O=R/'analysis_glm_regions'
A=O/'atlas'
S=A/'AAL3_source'
reg=json.loads((A/'AAL3_registration.json').read_text())
backup=O/'tables/pre_AAL3'
if not backup.exists():
 backup.mkdir()
 for p in (O/'tables').glob('*.tsv'):shutil.copyfile(p,backup/p.name)
 shutil.copyfile(O/'summary.json',backup/'summary.json')
base={'Precentral':'中央前回','Frontal_Sup_2':'额上回','Frontal_Mid_2':'额中回','Frontal_Inf_Oper':'额下回盖部','Frontal_Inf_Tri':'额下回三角部','Frontal_Inf_Orb_2':'额下回眶部','Rolandic_Oper':'中央沟岛盖','Supp_Motor_Area':'辅助运动区','Olfactory':'嗅皮层','Frontal_Sup_Medial':'额上回内侧部','Frontal_Med_Orb':'眶额内侧皮层','Rectus':'直回','OFCmed':'眶额皮层内侧部','OFCant':'眶额皮层前部','OFCpost':'眶额皮层后部','OFClat':'眶额皮层外侧部','Insula':'岛叶','Cingulate_Mid':'扣带回中部','Cingulate_Post':'扣带回后部','Hippocampus':'海马','ParaHippocampal':'海马旁回','Amygdala':'杏仁核','Calcarine':'距状沟皮层','Cuneus':'楔叶','Lingual':'舌回','Occipital_Sup':'枕上回','Occipital_Mid':'枕中回','Occipital_Inf':'枕下回','Fusiform':'梭状回','Postcentral':'中央后回','Parietal_Sup':'顶上小叶','Parietal_Inf':'顶下小叶','SupraMarginal':'缘上回','Angular':'角回','Precuneus':'楔前叶','Paracentral_Lobule':'旁中央小叶','Caudate':'尾状核','Putamen':'壳核','Pallidum':'苍白球','Heschl':'Heschl回（横颞回）','Temporal_Sup':'颞上回','Temporal_Pole_Sup':'颞极上部','Temporal_Mid':'颞中回','Temporal_Pole_Mid':'颞极中部','Temporal_Inf':'颞下回','Cerebellum_Crus1':'小脑Crus I','Cerebellum_Crus2':'小脑Crus II','Thal_AV':'丘脑前腹核','Thal_LP':'丘脑外侧后核','Thal_VA':'丘脑腹前核','Thal_VL':'丘脑腹外侧核','Thal_VPL':'丘脑腹后外侧核','Thal_IL':'丘脑板内核','Thal_Re':'丘脑Re核','Thal_MDm':'丘脑背内侧核内侧大细胞部','Thal_MDl':'丘脑背内侧核外侧小细胞部','Thal_LGN':'外侧膝状体','Thal_MGN':'内侧膝状体','Thal_PuI':'丘脑枕下部','Thal_PuM':'丘脑枕内侧部','Thal_PuA':'丘脑枕前部','Thal_PuL':'丘脑枕外侧部','ACC_sub':'前扣带皮层膝下部','ACC_pre':'前扣带皮层膝前部','ACC_sup':'前扣带皮层胼胝体上部','N_Acc':'伏隔核','VTA':'腹侧被盖区','SN_pc':'黑质致密部','SN_pr':'黑质网状部','Red_N':'红核','LC':'蓝斑','Raphe_D':'背侧中缝核','Raphe_M':'正中中缝核'}
labels={0:{'name_cn':'未分配','name_en':'Unassigned','hemisphere':'—'}}
for x in ET.parse(S / 'AAL3v1_1mm.xml').findall('.//label'):
    i=int(x.find('index').text)
    name=x.find('name').text
    side=name[-1] if name.endswith(('_L','_R')) else 'midline'
    stem=name[:-2] if side!='midline' else name
    if stem.startswith('Cerebellum_') and stem not in base:
        cn='小脑第'+stem.removeprefix('Cerebellum_').replace('_','–')+'小叶'
    elif stem.startswith('Vermis_'):
        cn='小脑蚓部第'+stem.removeprefix('Vermis_').replace('_','–')+'小叶'
    else:
        cn=base[stem]
    labels[i]={'name_cn':('左' if side=='L' else '右' if side=='R' else '')+cn,'name_en':name,'hemisphere':side,'source_index':i}
im=nib.load(A/'AAL3_space-MNI152NLin2009cAsym_res-2_dseg.nii.gz')
atlas=np.rint(im.get_fdata()).astype(int)
peaks=pd.read_csv(O/'tables/all_peak_labels.tsv',sep='\t')
old=peaks.drop(columns=[c for c in peaks if c.startswith('aal3_')]).copy()
overlap=[]
totals=[]
for run in ['run-'+r for r in CASE.runs]+['combined']:
    name=f'{CASE.subject}_task-{CASE.task}_'+run
    f=R/'derivatives/glm'/name
    z=nib.load(f/(name+'_contrast-taskVsRest_desc-zmap.nii.gz'))
    zz=z.get_fdata()
    clim=nib.load(f/(name+'_desc-fdr05_clusters.nii.gz'))
    cl=clim.get_fdata().astype(int)
    assert z.shape==im.shape and np.allclose(z.affine,im.affine) and np.allclose(clim.affine,im.affine)
    voxel=abs(np.linalg.det(im.affine[:3,:3]))
    for idx, row in peaks[peaks.run == run].iterrows():
        sign=1 if row.direction=='task>rest' else -1
        cid=int(row.cluster_id)
        m=cl==sign*cid
        xyz=[row.peak_x_mm,row.peak_y_mm,row.peak_z_mm]
        ijk=np.rint(nib.affines.apply_affine(np.linalg.inv(im.affine),xyz)).astype(int)
        lab=int(atlas[tuple(ijk)])
        peaks.loc[idx,'aal3_peak_label_id']=lab
        peaks.loc[idx,'aal3_peak_region_cn']=labels[lab]['name_cn']
        peaks.loc[idx,'aal3_peak_region_en']=labels[lab]['name_en']
        values,counts=np.unique(atlas[m],return_counts=True)
        entries=[]
        for value,count in sorted(zip(values.tolist(),counts.tolist()),key=lambda x:x[1],reverse=True):
         entry={'run':run,'cluster_id':cid,'direction':row.direction,'atlas':'aal3','label_id':value,'region_cn':labels[value]['name_cn'],'region_en':labels[value]['name_en'],'voxels':count,'volume_mm3':count*voxel,'percent_of_entire_cluster':100*count/row.voxels,'cluster_voxels':int(row.voxels)};entries.append(entry);overlap.append(entry)
        peaks.loc[idx,'aal3_assigned_percent']=100*np.count_nonzero(atlas[m])/row.voxels
        peaks.loc[idx,'aal3_top_overlap']='；'.join(x['region_cn']+f' {x["percent_of_entire_cluster"]:.1f}%' for x in entries[:4])
    for direction,sign in [('task>rest',1),('rest>task',-1)]:
     sig=cl*sign>0;n=int(sig.sum())
     for lab in np.unique(atlas[sig]):
      m=sig&(atlas==lab);idx=np.argwhere(m);peak=idx[np.argmax(sign*zz[m])];xyz=nib.affines.apply_affine(im.affine,peak)
      totals.append({'run':run,'direction':direction,'atlas':'aal3','label_id':int(lab),'region_cn':labels[int(lab)]['name_cn'],'region_en':labels[int(lab)]['name_en'],'significant_voxels':len(idx),'volume_mm3':len(idx)*voxel,'percent_of_direction_voxels':100*len(idx)/n,'regional_peak_z':zz[tuple(peak)],'regional_peak_x_mm':xyz[0],'regional_peak_y_mm':xyz[1],'regional_peak_z_mm':xyz[2]})
peaks['aal3_peak_label_id']=peaks.aal3_peak_label_id.astype(int)
pd.testing.assert_frame_equal(old,peaks[old.columns],check_exact=True)
ov=pd.read_csv(O/'tables/all_cluster_overlap.tsv',sep='\t')
ov=pd.concat([ov[ov.atlas!='aal3'],pd.DataFrame(overlap)],ignore_index=True)
tt=pd.read_csv(O/'tables/all_region_totals.tsv',sep='\t')
tt=pd.concat([tt[tt.atlas!='aal3'],pd.DataFrame(totals)],ignore_index=True)
peaks.to_csv(O/'tables/all_peak_labels.tsv',sep='\t',index=False)
ov.to_csv(O/'tables/all_cluster_overlap.tsv',sep='\t',index=False)
tt.to_csv(O/'tables/all_region_totals.tsv',sep='\t',index=False)
for run in ['run-'+r for r in CASE.runs]+['combined']:
 peaks[peaks.run==run].sort_values('peak_z',key=abs,ascending=False).to_csv(O/'tables'/f'{run}_peak_labels.tsv',sep='\t',index=False);ov[ov.run==run].to_csv(O/'tables'/f'{run}_cluster_overlap.tsv',sep='\t',index=False)
dic=pd.read_csv(A/'label_dictionary.tsv',sep='\t')
dic=pd.concat([dic[dic.atlas!='aal3'],pd.DataFrame([{'atlas':'aal3','label_id':i,**x} for i,x in labels.items()])],ignore_index=True)
dic.to_csv(A/'label_dictionary.tsv',sep='\t',index=False)
summ=json.loads((O/'summary.json').read_text())
summ['status']='AAL3_ADDED_QC_PENDING'
summ['strongest_peaks']=[peaks[(peaks.run==p['run'])&(peaks.cluster_id==p['cluster_id'])].iloc[0].to_dict() for p in summ['strongest_peaks']]
summ['atlases']=['cortical','subcortical','aal3']
summ['overlap_rows']=len(ov)
summ['AAL3_registration']='atlas/AAL3_registration.json'
(O/'summary.json').write_text(json.dumps(summ,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
pd.DataFrame(summ['strongest_peaks']).to_csv(O/'tables/strongest_peak_per_direction.tsv',sep='\t',index=False)
coverage=[]
for run in ['run-'+r for r in CASE.runs]+['combined']:
 for direction in ['task>rest','rest>task']:
  for key in ['cortical','subcortical','aal3']:
   g=tt[(tt.run==run)&(tt.direction==direction)&(tt.atlas==key)];n=int(g.significant_voxels.sum());un=int(g.loc[g.label_id==0,'significant_voxels'].sum());coverage.append({'run':run,'direction':direction,'atlas':key,'significant_voxels':n,'assigned_voxels':n-un,'unassigned_voxels':un,'unassigned_percent':100*un/n})
pd.DataFrame(coverage).to_csv(O/'tables/atlas_coverage_comparison.tsv',sep='\t',index=False)
cfg=json.loads((O/'configuration.json').read_text())
cfg['status']='AAL3_ADDED_QC_PENDING'
cfg['additional_AAL3']='AAL3v2 package with AAL3v1 label volume (166 source regions), Colin27 to patient T1 to MNI2009; separate atlas labeling, not merged with HO'
cfg['AAL3_registration']='atlas/AAL3_registration.json'
cfg['comparison_note']='Atlases have different definitions and coverage; region names and shares need not agree. No claim that unassigned always decreases.'
(O/'configuration.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'AAL3_ADDED','source_regions':166,'new_overlap_rows':len(overlap),'new_region_total_rows':len(totals),'strongest_peaks':[{k:p[k] for k in ['run','direction','peak_z','aal3_peak_region_cn']} for p in summ['strongest_peaks']],'coverage':coverage},ensure_ascii=False),flush=True)
