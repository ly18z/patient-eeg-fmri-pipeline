"""Summarize model candidates against registered anatomy; generate review artifacts."""
from case_config import CASE
from pathlib import Path
import json, gzip, base64, html, csv
import numpy as np,nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = CASE.root
OUT=ROOT/'derivatives/lesion'
QC=ROOT/'qc/lesion'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
stroke=read(OUT/'lesion_model_summary.json')
wmh=read(OUT/'model-truenet-wmh/model_result.json')
ref=nib.load(ROOT/f'bids/{CASE.subject}/anat/{CASE.subject}_T1w.nii.gz')
t1=ref.get_fdata(dtype=np.float32)
flair=nib.load(OUT/f'{CASE.subject}_space-T1w_desc-coreg_FLAIR.nii.gz').get_fdata(dtype=np.float32)
brain=nib.load(ROOT/f'derivatives/fmriprep/{CASE.subject}/anat/{CASE.subject}_desc-brain_mask.nii.gz').get_fdata()>.5
ho=nib.load(OUT/f'anatomy/{CASE.subject}_space-T1w_atlas-HarvardOxfordSubcortical_dseg.nii.gz').get_fdata().astype(np.uint8)
jhu=nib.load(OUT/f'anatomy/{CASE.subject}_space-T1w_atlas-JHUWhiteMatter_dseg.nii.gz').get_fdata().astype(np.uint8)
ho_en=read(OUT/'anatomy/atlas_labels.json')
jhu_en=read(OUT/'anatomy/jhu_labels.json')
def chinese(name):
 replacements={'Left':'左侧','Right':'右侧','Cerebral White Matter':'大脑白质','Cerebral Cortex':'大脑皮层','Lateral Ventrical':'侧脑室','Lateral Ventricle':'侧脑室','Thalamus':'丘脑','Caudate':'尾状核','Putamen':'壳核','Pallidum':'苍白球','Accumbens':'伏隔核','Brain-Stem':'脑干','Hippocampus':'海马','Amygdala':'杏仁核',
 'Posterior limb of internal capsule':'内囊后肢','Anterior limb of internal capsule':'内囊前肢','Retrolenticular part of internal capsule':'内囊豆状核后部','Anterior corona radiata':'前部放射冠','Superior corona radiata':'上部放射冠','Posterior corona radiata':'后部放射冠','Corticospinal tract':'皮质脊髓束参考区','External capsule':'外囊','Unclassified':'图谱未标记区'}
 side=''
 if name.endswith(' R'):side='右侧 ';name=name[:-2]
 if name.endswith(' L'):side='左侧 ';name=name[:-2]
 for a,b in replacements.items():name=name.replace(a,b)
 return side+name
ho_names={k:chinese(v) for k,v in ho_en.items()}
jhu_names={k:chinese(v) for k,v in jhu_en.items()}
models=[]
masks=[]
flags=np.zeros(ref.shape,np.uint8)
voxel=abs(np.linalg.det(ref.affine[:3,:3]))
bg_ids=[11,12,13,26,50,51,52,58]
for index, (key, name, path, probpath) in enumerate([('baseline', 'ATLAS T1卒中模型', OUT / f'model-baseline/{CASE.subject}_space-T1w_model-baseline_desc-braincandidate_mask.nii.gz', OUT / f'model-baseline/{CASE.subject}_space-T1w_model-baseline_desc-lesion_probseg.nii.gz'), ('synth', 'SynthStroke卒中模型', OUT / f'model-synth/{CASE.subject}_space-T1w_model-synth_desc-braincandidate_mask.nii.gz', OUT / f'model-synth/{CASE.subject}_space-T1w_model-synth_desc-lesion_probseg.nii.gz'), ('wmh', 'TrUE-Net白质高信号模型', OUT / f'model-truenet-wmh/{CASE.subject}_space-T1w_model-TrUENet_desc-WMHcandidate_mask.nii.gz', OUT / f'model-truenet-wmh/{CASE.subject}_space-T1w_model-TrUENet_desc-WMH_probseg.nii.gz')]):
    im=nib.load(path)
    mask=im.get_fdata()>.5
    prob=nib.load(probpath).get_fdata(dtype=np.float32)
    masks.append(mask)
    flags[mask]|=(1<<index)
    labs,n=ndimage.label(mask,structure=ndimage.generate_binary_structure(3,1))
    components=[]
    for component in range(1,n+1):
     ijk=np.argwhere(labs==component);sel=labs==component
     def regions(a,names):
      ids,counts=np.unique(a[sel],return_counts=True);order=np.argsort(counts)[::-1]
      return [dict(label=int(ids[o]),name=names.get(str(ids[o]),'图谱未标记/边界区'),voxels=int(counts[o]),fraction=float(counts[o]/sel.sum())) for o in order[:4]]
     centroid=nib.affines.apply_affine(ref.affine,ijk.mean(axis=0))
     components.append(dict(component_id=component,voxels=int(sel.sum()),volume_ml=float(sel.sum()*voxel/1000),
      centroid_native_mm=centroid.tolist(),peak_model_probability=float(prob[sel].max()),HO_regions=regions(ho,ho_names),JHU_regions=regions(jhu,jhu_names)))
    components.sort(key=lambda x:x['voxels'],reverse=True)
    record=dict(key=key,name=name,candidate_voxels=int(mask.sum()),candidate_volume_ml=float(mask.sum()*voxel/1000),
     max_model_probability_in_brain=float(prob[brain].max()),basal_ganglia_overlap_ml=float(np.sum(mask&np.isin(ho,bg_ids))*voxel/1000),
     atlas_white_matter_overlap_ml=float(np.sum(mask&np.isin(ho,[2,41]))*voxel/1000),components=components)
    models.append(record)
    inv=np.linalg.inv(ref.affine)
    zs=[5,10,15,20,25,30,35,40]
    ks=[int(np.clip(round(nib.affines.apply_affine(inv,[0,0,z])[2]),0,ref.shape[2]-1)) for z in zs]
    fig,axes=plt.subplots(2,8,figsize=(20,7),constrained_layout=True)
    for row,(arr,modality) in enumerate([(t1,'T1'),(flair,'FLAIR')]):
     lo,hi=np.percentile(arr[brain],[1,99.5])
     for col,k in enumerate(ks):
      ax=axes[row,col];ax.imshow(arr[:,:,k].T,origin='lower',cmap='gray',vmin=lo,vmax=hi)
      sl=mask[:,:,k].T
      if sl.any():ax.imshow(np.ma.masked_where(~sl,sl),origin='lower',cmap=matplotlib.colors.ListedColormap(['#ff9638']),alpha=.7,vmin=0,vmax=1)
      bg=np.isin(ho[:,:,k].T,bg_ids)
      if bg.any():ax.contour(bg,levels=[.5],colors=['#63d49c'],linewidths=.4)
      ax.axis('off');ax.set_title(f'{modality} | k={k}',fontsize=9)
    fig.suptitle(f'{key}: {record["candidate_volume_ml"]:.3f} mL candidates | orange=candidate, green=basal ganglia reference',fontsize=13)
    fig.savefig(QC/f'model-{key}_T1_FLAIR_candidates.png',dpi=150)
    plt.close(fig)
write(OUT/'candidate_localization.json',dict(status='GENERATED_REVIEW_PENDING',models=models,
 interpretation='Atlas overlaps are reference localization; WMH candidates do not establish infarction, hemorrhage or functional tract damage.',
 atlas_note='HO source is FreeSurfer-reencoded aseg IDs; JHU is a population region atlas. Both are warped references, not patient-specific nuclear/tract segmentations.'))
micro_note=''
if models[2]['components']:
 single=sum(c['voxels']==1 for c in models[2]['components']);largest=models[2]['components'][0]['voxels']
 micro_note=f'白质模型共有{len(models[2]["components"])}个连通候选，其中{single}个仅含1个重采样体素，最大区域只有{largest}个体素。部分标签与脑室边界参考区重叠；这类极小标签可能受部分容积、配准及模型误检影响，不能直接认定为独立病灶。'
 fig,axes=plt.subplots(2,6,figsize=(16,6),constrained_layout=True)
 probability=nib.load(OUT/f'model-truenet-wmh/{CASE.subject}_space-T1w_model-TrUENet_desc-WMH_probseg.nii.gz').get_fdata()
 labs,_=ndimage.label(masks[2],structure=ndimage.generate_binary_structure(3,1))
 for group,c in enumerate(models[2]['components'][:2]):
  loc=np.where(labs==c['component_id'],probability,-1);i,j,k=np.unravel_index(np.argmax(loc),loc.shape)
  for row,(arr,modality) in enumerate([(t1,'T1'),(flair,'FLAIR')]):
   lo,hi=np.percentile(arr[brain],[1,99.5])
   for view,(im,ma,cx,cy) in enumerate([(arr[:,:,k].T,masks[2][:,:,k].T,i,j),(arr[:,j,:].T,masks[2][:,j,:].T,i,k),(arr[i,:,:].T,masks[2][i,:,:].T,j,k)]):
    ax=axes[row,group*3+view];ax.imshow(im,origin='lower',cmap='gray',vmin=lo,vmax=hi)
    ax.imshow(np.ma.masked_where(~ma,ma),origin='lower',cmap=matplotlib.colors.ListedColormap(['#ff9638']),alpha=.7,vmin=0,vmax=1)
    ax.set_xlim(cx-22,cx+22);ax.set_ylim(cy-22,cy+22);ax.axis('off');ax.set_title(f'C{c["component_id"]}: {c["voxels"]} vox | {modality}\n'+['axial','coronal','sagittal'][view],fontsize=10)
 fig.suptitle('Largest two WMH candidates: local magnification for review, not confirmed lesions',fontsize=13)
 fig.savefig(QC/'WMH_candidate_focus.png',dpi=160);plt.close(fig)
rows=[]
for m in models:
 for c in m['components']:
  rows.append(dict(model=m['key'],component=c['component_id'],voxels=c['voxels'],volume_ml=c['volume_ml'],
   native_x_mm=c['centroid_native_mm'][0],native_y_mm=c['centroid_native_mm'][1],native_z_mm=c['centroid_native_mm'][2],
   HO_reference='; '.join(x['name']+f' ({x["fraction"]:.0%})' for x in c['HO_regions']),
   JHU_reference='; '.join(x['name']+f' ({x["fraction"]:.0%})' for x in c['JHU_regions'])))
with (OUT/'candidate_components.tsv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['model','component','voxels','volume_ml','native_x_mm','native_y_mm','native_z_mm','HO_reference','JHU_reference'],delimiter='\t');w.writeheader();w.writerows(rows)
def packed(a):return base64.b64encode(gzip.compress(np.asarray(a,np.uint8).tobytes(order='C'),mtime=0)).decode()
def gray(a):
 lo,hi=np.percentile(a[brain],[1,99.5]);return np.rint(np.clip((a-lo)/(hi-lo),0,1)*255).astype(np.uint8)
initial=dict(x=ref.shape[0]//2,y=ref.shape[1]//2,z=int(nib.affines.apply_affine(np.linalg.inv(ref.affine),[0,0,20])[2]))
if models[2]['components']:
 largest=models[2]['components'][0];point=np.rint(nib.affines.apply_affine(np.linalg.inv(ref.affine),largest['centroid_native_mm'])).astype(int)
 initial=dict(zip(['x','y','z'],map(int,point)))
payload=dict(shape=list(ref.shape),affine=ref.affine.tolist(),t1=packed(gray(t1)),flair=packed(gray(flair)),flags=packed(flags),ho=packed(ho),jhu=packed(jhu),initial=initial,
 hoLabels=ho_names,jhuLabels=jhu_names,modelInfo=[m['name']+f'：默认规则下脑内候选 {m["candidate_volume_ml"]:.3f} mL，{len(m["components"])} 个6邻接连通区域。' for m in models])
template=(Path(__file__).with_name('lesion_viewer_template.html')).read_text(encoding='utf-8')
(ROOT/'候选病变影像查看.html').write_text(template.replace('__DATA_JSON__',json.dumps(payload,ensure_ascii=False,separators=(',',':'))),encoding='utf-8')
summary_table='| 模型 | 默认脑内候选体积(mL) | 基底节参考区重叠(mL) | 白质参考区重叠(mL) |\n|---|---:|---:|---:|\n'+'\n'.join(f'| {m["name"]} | {m["candidate_volume_ml"]:.3f} | {m["basal_ganglia_overlap_ml"]:.3f} | {m["atlas_white_matter_overlap_ml"]:.3f} |' for m in models)
finding=('两套T1卒中模型均未分出脑内候选区域。' if not any(masks[i].any() for i in [0,1]) else 'T1模型产生了需要复核的卒中候选区域。')+f'TrUE-Net在T1+FLAIR上分出 {models[2]["candidate_volume_ml"]:.3f} mL 白质高信号候选。'
note=f'''# 病灶候选定位：模型运行结果

病程、卒中类型和影像报告以本次病例资料为准。本轮完成自动候选分割、T1/FLAIR对照及解剖参考定位；没有将自动结果当作临床确诊病灶，也没有手工修改候选掩膜。

## 主要结果

{finding}

{micro_note}

{summary_table}

“未检出”表示模型在模型没有输出脑内阳性标签时，也不等于排除既往卒中、小病灶或陈旧出血残留。白质高信号候选不直接等于脑梗死、脑出血或皮质脊髓束损伤。基底节/白质栏是人群图谱配准后的几何重叠，不是被确诊损伤的组织体积。

[打开患者T1/FLAIR交互核查页](候选病变影像查看.html)。可切换模型、T1/FLAIR背景、切片和透明度。绿色轮廓为基底节参考位置，橙色才是所选模型的候选标签。

## 流程与固定参数

1. 使用本次T1原始网格及fMRIPrep的T1偏置场校正、脑掩膜。本次FLAIR另做N4偏置场校正，再以互信息刚性配准到T1，保留原始输入。
2. SynthStroke baseline使用真实ATLAS T1卒中数据训练的公开模型；synth使用合成多组织模型。网络结构按官方配置加载；推理采用官方支持的128³滑窗、50%重叠、高斯logit融合，不用TTA。输入按作者test.py进行RAS、1 mm采样、直方图均衡和整体强度标准化；在空间变换前保留正确仿射元数据。反变换logit后softmax与argmax得到原生网格标签，之后仅去掉现有T1脑掩膜外的候选。
3. TrUE-Net MWSC采用配对T1+FLAIR官方预训练网络：头骨外信号置零；使用作者的脑区裁剪、强度标准化及三平面尺寸变换，平均三个平面的预测概率，固定阈值>0.5并限定脑掩膜。没有根据结果调整阈值或重新训练模型。
4. Harvard-Oxford皮层下图谱及其MNI152NLin6Asym参考图像另行配准到患者T1，不把MNI6与此前GLM的MNI2009空间混为一谈。FreeSurfer发布副本的标签采用aseg编号，已显式解码并核查。JHU白质区使用同一空间变换作参考定位。
5. 连通区域按6邻接定义；原始候选和所有大小的区域均保存，没有为得到阳性结果删选或放大区域。模型分数是分类输出，不是GLM的p值，不套用FDR或Bonferroni。

## 可核查图

![ATLAS模型](qc/lesion/model-baseline_T1_FLAIR_candidates.png)

![合成模型](qc/lesion/model-synth_T1_FLAIR_candidates.png)

![白质高信号模型](qc/lesion/model-wmh_T1_FLAIR_candidates.png)

![较大的两个微小候选局部放大](qc/lesion/WMH_candidate_focus.png)

## 定位文件

- `derivatives/lesion/candidate_components.tsv`：每个候选连通区域的体积、原生物理坐标重心、HO与JHU参考区组成。重心不一定落在区域内部，不能当作病变的单一中心或手术坐标。
- `derivatives/lesion/candidate_localization.json`：完整定位与模型分布统计。
- `derivatives/lesion/model-baseline/`、`model-synth/`：未裁剪概率图、原始多类别标签、脑外去除前后掩膜。synth原始输出包含脑外标签，已单独保留并从脑内结果排除。
- `derivatives/lesion/model-truenet-wmh/`：三个平面概率图、融合概率图、固定阈值WMH候选掩膜。
- `derivatives/lesion/anatomy/`：图谱、标签表、变换与配准记录；`qc/lesion/`为多序列和候选叠加检查图。
- `tools/lesion_models/`保存公开模型权重、源码提交号与校验元数据；不包含向服务端上传患者影像的步骤。

## 解释边界

当前脚本没有载入人工病灶真值，无法计算该患者分割的真实Dice或灵敏度。未检出不作为“没有卒中”的诊断依据；WMH阳性也不单独决定病因。小的深部病变、陈旧出血残留及特定病程可能不被T1模型识别。此处的组织分区与JHU参考区也不替代患者实际的扩散束追踪。若需要确认病因与最终病灶边界，应结合原始完整多序列影像和临床判读。

公开方法：[SynthStroke代码与模型](https://github.com/liamchalcroft/SynthStroke)、[ATLAS T1预训练模型](https://huggingface.co/liamchalcroft/synthstroke-baseline)、[TrUE-Net](https://github.com/v-sundaresan/truenet)。本轮未运行面向急性DWI/ADC的DeepISLES，也未将MS专用模型直接当作卒中模型。
'''
(ROOT/'病灶候选定位说明.md').write_text(note,encoding='utf-8')
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
body='<h1>病灶候选定位：T1＋FLAIR模型结果</h1><p>'+html.escape(finding)+'</p><p><a href="候选病变影像查看.html">打开原生T1/FLAIR交互核查页 →</a></p>'
body+='<p>'+html.escape(micro_note)+'</p>'
body+=table(['模型','脑内候选(mL)','基底节参考重叠(mL)','白质参考重叠(mL)'],[[m['name'],f'{m["candidate_volume_ml"]:.3f}',f'{m["basal_ganglia_overlap_ml"]:.3f}',f'{m["atlas_white_matter_overlap_ml"]:.3f}'] for m in models])
body+='<p class="note">默认规则未检出不能排除既往卒中；白质高信号候选不直接等于卒中病灶。参考区重叠是空间定位估计，不是确诊的受损组织体积。</p>'
for m in models:
 body+='<section><h2>'+html.escape(m['name'])+'</h2><p>橙色：模型候选；绿色：基底节参考轮廓。上下两排分别是同一原生网格的T1与FLAIR。</p><img src="qc/lesion/model-'+m['key']+'_T1_FLAIR_candidates.png">'
 components=m['components'][:15]
 if components:
  body+=table(['区域','体积(mL)','原生重心 x,y,z(mm)','HO参考','JHU参考'],[[str(c['component_id']),f'{c["volume_ml"]:.3f}',', '.join(f'{x:.1f}' for x in c['centroid_native_mm']),'; '.join(x['name'] for x in c['HO_regions'][:2]),'; '.join(x['name'] for x in c['JHU_regions'][:2])] for c in components])
 else:body+='<p>默认分类规则下没有脑内候选区域。</p>'
 body+='</section>'
body+='<section><h2>微小候选的局部核查</h2><p>下面只是局部放大；候选大小见本次连通区域表，不能根据放大后的视觉面积认定病变大小或性质。</p><img src="qc/lesion/WMH_candidate_focus.png"></section>'
body+='<section><h2>配准与解剖参考</h2><p>T1/FLAIR已配准到患者原始T1网格；皮层下图谱使用其匹配的MNI6模板单独配准。图谱不是病灶标签。</p><img src="qc/lesion/atlas_basalGanglia_overlay.png"><img src="qc/lesion/atlas_whiteMatter_overlay.png"></section>'
body+='<section><h2>如何解释</h2><p>病程与病因以本次病例资料为准。本轮是自动候选定位，没有临床真值或人工确认的最终病灶。结果不足以单独判定梗死、出血、责任病灶或运动通路损伤。</p><p>模型概率不是统计检验p值；本轮没有用FDR或Bonferroni筛选模型候选，也没有为取得阳性结果调低阈值。极小连通区域也完整保存在文件中。</p><p><a href="病灶候选定位说明.md">完整中文说明</a> · <a href="derivatives/lesion/candidate_components.tsv">全部候选区域表</a> · <a href="derivatives/lesion/candidate_localization.json">定位统计</a> · <a href="derivatives/lesion/lesion_model_summary.json">T1模型记录</a></p></section>'
style='body{font:16px/1.75 "Microsoft YaHei",sans-serif;background:#f1f5f8;color:#203546;margin:0}main{max-width:1450px;margin:auto;padding:32px}h1{font-size:30px}h2{font-size:23px}section{background:white;padding:22px;margin:24px 0;border-radius:12px}img{width:100%;height:auto;margin:15px 0}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{padding:10px;text-align:left;border-bottom:1px solid #d9e3ea}th{background:#dfedf3}a{color:#087887}.note{background:#fff3db;padding:16px;border-radius:8px}'
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>病灶候选定位报告</title><style>'+style+'</style><main>'+body+'</main></html>'
(ROOT/'病灶候选定位报告.html').write_text(page,encoding='utf-8')
print(json.dumps(dict(status='REPORT_GENERATED_REVIEW_PENDING',models=[{k:m[k] for k in ['key','candidate_volume_ml','basal_ganglia_overlap_ml','atlas_white_matter_overlap_ml']} for m in models],components=len(rows)),ensure_ascii=False),flush=True)
