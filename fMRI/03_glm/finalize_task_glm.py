"""Render saved GLM maps and summarize the actual runs and event tables."""
from case_config import CASE
import html
import json
import numpy as np
import nibabel as nib
from nilearn import plotting

ROOT = CASE.root
OUT = ROOT/'derivatives/glm'
QC = ROOT/'qc/glm'
summary = json.loads((OUT/'glm_summary.json').read_text(encoding='utf-8'))
anat = ROOT/f'derivatives/fmriprep/{CASE.subject}/anat/{CASE.subject}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz'
mask = nib.load(next(OUT.glob('*desc-common_mask.nii.gz'))).get_fdata() > .5
models = [(f'{CASE.subject}_task-{CASE.task}_run-{r["model"]["run"]}', 'Run '+r['model']['run'], r['statistics']) for r in summary['runs']]
models += [(f'{CASE.subject}_task-{CASE.task}_combined', 'Combined', summary['combined'])]
for label, title, stats in models:
    zmap = nib.load(OUT/label/f'{label}_contrast-taskVsRest_desc-zmap.nii.gz')
    threshold = stats['fdr05']['z_threshold']
    if threshold is None:
        threshold = float(np.max(np.abs(zmap.get_fdata()[mask]))+1)
    for mode, cuts, suffix in [('z', [-10,0,10,20,30,40,50,60,70], 'axial'), ('ortho', (0,-20,50), 'ortho')]:
        display = plotting.plot_stat_map(zmap, bg_img=str(anat), display_mode=mode, cut_coords=cuts,
            threshold=threshold, cmap='cold_hot', symmetric_cbar=True, vmax=10,
            title=title+' | two-sided voxel FDR q<0.05')
        display.savefig(str(QC/f'{label}_fdr05_{suffix}.png'), dpi=160)
        display.close()

def table(headers, rows):
    return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'

rows=[]
for record in summary['runs']:
    m=record['model']
    rows.append([m['run'], m['source_series'], m['n_volumes'], m.get('TR_seconds', CASE.acquisition(CASE.bold(m['run']))[1]),
                 m['rank'], m['nominal_residual_dof'], m['spike_count']])
body='<h1>任务GLM分析</h1><p>'+html.escape(CASE.subject+' · '+CASE.task_settings['description'])+'</p>'
body+=table(['Run','源序列','体积数','TR（s）','设计秩','名义残差维度','异常帧回归项'], rows)
body+='<p>逐run拟合；合并使用效应及方差的逆方差固定效应。主要推断为双侧体素FDR q&lt;0.05，连通簇仅作空间分组。</p>'
for label,title,stats in models:
    body+='<section><h2>'+title+'</h2>'+table(['方向','FDR体素数'], [('正向',stats['fdr05']['positive_voxels']),('负向',stats['fdr05']['negative_voxels'])])
    for suffix in ['axial','ortho']:
        body+=f'<img src="qc/glm/{label}_fdr05_{suffix}.png">'
    body+=f'<p><a href="derivatives/glm/{label}/{label}_desc-fdr05_clusters.tsv">完整簇表</a></p></section>'
body+='<section><h2>任务时序与模型</h2><p>'+html.escape(summary['configuration']['timing_source'])+'</p>'
for record in summary['runs']:
    run=record['model']['run']
    label=f'{CASE.subject}_task-{CASE.task}_run-{run}'
    import pandas as pd
    events=pd.read_csv(OUT/label/f'{label}_events.tsv',sep='\t')
    body+='<h3>Run '+run+'</h3>'+table(['起点（s）','时长（s）','条件'], events[['onset','duration','trial_type']].values.tolist())
    body+=f'<img src="qc/glm/run-{run}_timing_motion.png"><img src="qc/glm/run-{run}_design_matrix.png">'
body+='<p>5 mm平滑；SPM HRF加时间导数；24项头动、5项combined aCompCor、fMRIPrep余弦漂移及异常帧回归；AR(1)。不重复高通。</p>'
body+='<p><a href="derivatives/glm/glm_summary.json">完整计算摘要</a> · <a href="derivatives/glm/analysis_configuration.json">实际模型配置</a></p></section>'
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>GLM分析报告</title><style>body{font:16px sans-serif;max-width:1150px;margin:30px auto;line-height:1.7}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:7px}</style>'+body+'</html>'
(ROOT/'GLM分析报告.html').write_text(page,encoding='utf-8')
(ROOT/'GLM分析结果说明.md').write_text('# 任务GLM\n\n'+CASE.task_settings['description']+'\n\n各run的实际体积数、TR、事件、设计矩阵与统计结果见GLM分析报告及derivatives/glm。\n',encoding='utf-8')
print(ROOT/'GLM分析报告.html')
