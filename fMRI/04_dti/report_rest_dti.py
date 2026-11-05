from case_config import CASE
import json, html, base64
import numpy as np
R = CASE.root
O=R/'analysis_rest_dti'
rest=json.loads((O/'rest/summary.json').read_text(encoding='utf-8'))
dti=json.loads((O/'dti/summary.json').read_text(encoding='utf-8'))
def esc(x):return html.escape(str(x))

def figure(name, caption=''):
    p=O/'qc'/name
    return '<figure><img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'" alt="'+esc(caption or name)+'">'+('<figcaption>'+esc(caption)+'</figcaption>' if caption else '')+'</figure>'
def table(headers,rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def attach(rel,label,name=None):
    p=O/rel;mime='application/json' if p.suffix=='.json' else 'text/tab-separated-values;charset=utf-8'
    return '<a download="'+esc(name or p.name)+'" href="data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode()+'">'+esc(label)+'</a>'
mat=np.array(rest['correlation_matrix'])
gsr=np.array(rest['correlation_GSR_sensitivity'])
primary=[x for x in dti['tractography'] if x['FA_stop_threshold']==.15]
body='<header><h1>静息态与扩散分析</h1><p>静息态与DWI · 本次配置数据</p><nav><a href="#rest">运动网络连接</a><a href="#dti">DTI与白质通路</a><a href="#methods">方法与质控</a></nav></header>'
body+='<section id="rest"><h2>1．静息态运动网络</h2><p>下表展示控制头动、组织噪声、漂移和异常帧后的运动／感觉参考区相关系数。结果为单例描述，不能据此判定是否偏离正常。</p>'
body+=table(['左右同源参考区','主要分析 r','加入全局信号回归 r'],[(name,f'{mat[i,j]:.3f}',f'{gsr[i,j]:.3f}') for i,j,name in [(0,1,'中央前回'),(2,3,'辅助运动区'),(4,5,'中央后回')]])
body+=figure('rest_motor_FC_matrices.png','r表示时间序列的同步程度，不表示因果关系或实际白质纤维连接。右图为处理方法敏感性对照。')
body+='<h3>区域如何定义</h3><p>使用Harvard–Oxford解剖图谱，经患者T1映射到分析空间，并限定灰质概率≥0.2。中央前回参考区包含较广的运动皮层，不是个体手区；本轮未单独划出运动前区，也未依据任务显著性选择区域。</p>'
body+=table(['参考区','原生空间覆盖','有效信号覆盖','纳入体素'],[(x['name'],f'{100*x["native_BOLD_coverage_fraction"]:.1f}%',f'{100*x.get("effective_signal_coverage_fraction",x["standard_coverage_fraction"]):.1f}%',x.get('effective_signal_voxels',x['standard_valid_voxels'])) for x in rest['coverage']])
body+=figure('motor_ROIs_MNI.png','解剖ROI示意；脑区边界是图谱映射近似，不是人工逐点确认。')
body+='<details><summary>时间曲线与探索性滞后检查</summary>'+figure('rest_ROI_timeseries.png','红线标记异常帧，连接估计未纳入这些观测。')+figure('rest_interhemispheric_lag.png','三组左右同源区的最大相关滞后见本次计算摘要；它不是任务HRF峰值延迟估计，也不能排除其他脑区的血流动力学异常。')+'</details>'
body+='<details><summary>种子到全脑的连接图</summary><p>图中仅显示|r|≥0.2以便观察，不是统计显著性阈值；未进行体素水平显著性检验。保存的Fisher z图为atanh(r)变换，不是GLM的Z统计图。</p>'
for i in range(1,5):body+=figure(f'rest_seed_{i}.png')
body+='</details><p class="links">'+attach('rest/motor_FC_correlation.tsv','连接矩阵')+attach('rest/ROI_timeseries.tsv','区域时间序列')+attach('rest/interhemispheric_lag_exploratory.tsv','滞后检查数据')+'</p></section>'
body+='<section id="dti"><h2>2．DTI与白质通路</h2><p>完成MP-PCA去噪、FSL eddy头动／涡流校正，以及加权最小二乘张量拟合。输出FA、MD、AD、RD；磁敏感畸变尚未校正。</p>'
body+=table(['质量指标','结果'],[['原始扩散采样','采样参数见本次DWI头信息与梯度表'],['梯度核对',f'DICOM方向转换后与导出方向的最小绝对点积 {dti["gradient_validation"]["minimum_absolute_dot_with_ordered_DICOM_voxel_gradients"]:.6f}'],['配准脑掩膜Dice',f'{dti["registration_mask_Dice"]:.3f}'],['白质参考范围FA中位数',f'{dti["WM_FA_median"]:.3f}'],['白质参考范围MD中位数',f'{dti["WM_MD_median_mm2_s"]*1000:.3f} ×10⁻³ mm²/s'],['白质归一化拟合RMSE中位数',f'{dti["WM_normalized_fit_RMSE_median"]:.3f}']])
body+=figure('dti_FA.png','FA描述扩散方向性，不能直接等同于白质完整性。')+figure('dti_MD.png','MD图以10⁻³ mm²/s显示。')
body+='<h3>解剖参考区的扩散指标</h3><p>以下是映射到患者扩散空间的JHU参考区域，限制在脑掩膜与白质概率≥0.3范围内；不是病灶分割。</p>'
body+=table(['参考区域','体素数','平均FA','平均MD（×10⁻³ mm²/s）'],[(x['region'],x['voxel_count'],f'{x["FA_mean"]:.3f}' if x['FA_mean'] is not None else '无有效体素',f'{x["MD_mean"]*1000:.3f}' if x['MD_mean'] is not None else '—') for x in dti['reference_regions']])
body+='<h3>皮质脊髓束候选追踪</h3><p>从内囊后肢参考区播种，要求线条同时经过同侧脑脚及中央前回附近，并排除对侧中央前回。仅保留脑脚至中央前回的40–250 mm分析段。主要停止阈值FA≥0.15；另以0.10做敏感性对照。单张量模型不能解决交叉纤维，线条数量不是轴突数量。</p>'
body+=table(['侧别','FA停止阈值','播种点','满足约束的线条','覆盖白质平均FA','状态'],[(x['side'],f'{x["FA_stop_threshold"]:.2f}',x['seeds'],x['qualified_streamlines'],f'{x["FA_mean"]:.3f}' if x['qualified_streamlines'] else '—','候选' if x['qualified_streamlines'] else '未恢复，不等于通路缺失') for x in dti['tractography']])
if (O/'qc/dti_CST_candidates_3D.png').exists():body+=figure('dti_CST_candidates_3D.png','左橙／右青；患者扩散数据生成的候选线条。')
if (O/'qc/dti_CST_on_T1.png').exists():body+=figure('dti_CST_on_T1.png','候选束密度范围映射至患者T1；左橙／右青。')
if (O/'qc/dti_candidate_along_FA.png').exists():body+=figure('dti_candidate_along_FA.png','归一化长度的描述性FA曲线；阴影是线条间四分位范围，不是置信区间，也不保证左右每个节点解剖对应。')
body+='<details><summary>配准与参考区质控</summary>'+figure('dti_registration_edges.png','患者T1上的b0边缘；整体配准不能替代局部磁敏感畸变校正。')+figure('dti_JHU_reference.png','JHU参考区域在扩散图上的位置。')+'</details>'
body+='<p class="links">'+attach('dti/JHU_reference_region_metrics.tsv','参考区指标')+attach('dti/tractography_summary.tsv','追踪结果表')
if (O/'dti/along_candidate_tract_FA.tsv').exists():body+=attach('dti/along_candidate_tract_FA.tsv','沿候选束FA数据')
body+='</p></section>'
prep=dti['preprocessing']
motion=prep.get('movement_RMS_mm',{})
outliers=prep.get('outlier_slices',{})
body+='<section id="methods"><h2>3．方法与使用范围</h2>'+table(['静息态指标','结果'],[('TR（s）',rest['TR_seconds']),('体积数',rest['frames']),('保留帧数',rest['retained_frames']),('残差维度',rest['denoising']['residual_dimension']),('频段',rest['denoising']['filter'])])
body+='<p>实际异常帧标准见配置和去噪设计矩阵。滤波及混杂回归后的残差维度与保留帧数见本次摘要；保留帧数不能视为独立样本数。预处理实际校正步骤见本次fMRIPrep记录。</p>'
body+=table(['项目','实际设置'],[['静息态观测',f'{rest["frames"]}帧中排除{rest["censored_frames"]}帧，保留{rest["retained_frames"]}帧；观测合计{rest["retained_observation_seconds"]}秒，时间轴仍保留原间隙。'],['静息态去噪','24项头动＋5项combined aCompCor＋异常帧指示项＋截距；'+rest['denoising']['filter']+'；不重复加入fMRIPrep余弦漂移项。'],['静息态模型',f'设计秩{rest["denoising"]["design_rank"]}，残差维数{rest["denoising"]["residual_dimension"]}；不进行空间平滑；主分析不做全局信号回归。'],['无效信号位置',f'排除{rest["signal_mask"]["initial_voxels"]-rest["signal_mask"]["retained_voxels"]}个零值、极低均值或恒定体素；最终脑掩膜{rest["signal_mask"]["retained_voxels"]}体素。'],['扩散校正','MP-PCA patch半径2；eddy二次涡流模型、5轮估计、体积间运动校正和异常层替换；使用eddy旋转后的梯度。'],['扩散采集参数',f'相位编码轴{prep["acquisition_parameters"]["PE_axis"]}；读出时间{prep["acquisition_parameters"]["readout_seconds"]:.6f}秒。极性及来源见预处理记录；没有运行TOPUP或使用外部场图。'],['扩散运动',f'绝对RMS均值／最大值：{motion.get("absolute_mean",float("nan")):.3f}／{motion.get("absolute_max",float("nan")):.3f} mm；相对RMS均值：{motion.get("relative_mean",float("nan")):.3f} mm。'],['扩散异常层',f'{outliers.get("count","未记录")}／{outliers.get("total","未记录")}个层观测被标记，约{100*outliers.get("fraction",0):.2f}%。'],['空间对应','图谱MNI6→患者T1；运动ROI再映射至MNI2009静息态空间。扩散b0到T1使用刚性互信息配准，未以非线性配准冒充畸变校正。']])
body+='<p>当前病灶和患侧尚未确认，均按左／右报告。单例结果用于探索，不能据相关系数、FA、左右差异或追踪失败直接判断损伤程度和临床功能。</p><p class="links">'+attach('rest/summary.json','静息态完整记录','rest_summary.json')+attach('dti/summary.json','扩散完整记录','dti_summary.json')+'</p></section>'
style='body{margin:0;background:#f2f6f9;color:#213746;font:15px/1.8 "Microsoft YaHei",sans-serif}main{max-width:1400px;margin:auto;padding:28px}h1{font-size:30px;margin:0}h2{font-size:24px;margin:0 0 12px}h3{font-size:18px}header p,figcaption{color:#607687}nav,.links{display:flex;flex-wrap:wrap;gap:18px}a{color:#087c8c}section{background:white;padding:24px;border-radius:12px;margin:24px 0}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:11px;border-bottom:1px solid #dce7ee;text-align:left;vertical-align:top}th{background:#eaf2f6}figure{margin:20px 0}img{width:100%;height:auto;border:1px solid #e0e9ef}figcaption{font-size:13px}details{margin:18px 0}summary{cursor:pointer;color:#176c7c;font-weight:bold;padding:9px;background:#eef5f8}@media(max-width:700px){main{padding:12px}section{padding:15px}}'
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>静息态与扩散分析</title><style>'+style+'</style><main>'+body+'</main></html>'
(O/'静息态与扩散分析报告.html').write_text(page,encoding='utf-8')
note=f'''# 静息态与扩散分析

打开同目录“静息态与扩散分析报告.html”。该报告与原总报告分开。

- rest/：连接矩阵、时间序列、保留帧索引、种子连接图、去噪BOLD、配置与结果。
- anatomy/：运动图谱及其个体映射，覆盖检查。中央前回不是手区专用ROI。
- dti/：FA/MD/AD/RD、张量拟合误差、扩散预处理、配准、解剖参考区指标及候选追踪。
- dti/tractography/：候选TRK与密度图。参数和约束见dti/summary.json。
- qc/：可核查图；计算代码见本仓库；work/：模板、图谱及配准资源。

静息态保留{rest['retained_frames']}帧，去噪NIfTI仍有{rest['frames']}帧，其中排除帧置零；再次分析务必同时使用rest/retained_volume_indices_0based.txt，不能把零帧当作正常观测。

扩散完成头动/涡流校正，未完成磁敏感畸变校正。相位编码极性没有被当作真实已知元数据；无外部场图的eddy内部参数约定与估计读出时间完整保存在dti/preprocessing/configuration.json。单b0及3mm层间采样是限制。未进行常规多壳层NODDI。

主连接图是描述性相关，展示阈值不是统计显著性；lag结果不是任务HRF延迟。FA和线条数不能直接诊断损伤。没有确认病灶侧别、正常对照或独立临床验证。

原总报告和既有GLM输入结果保留，未汇入本次结果。
'''
(O/'结果说明.md').write_text(note,encoding='utf-8')
original=R/'MRI分析与影像整合报告.html'
print(json.dumps({'status':'REPORT_READY_FOR_VISUAL_QA','report_bytes':len(page),'main_report_unchanged':True,'left_right_precentral_r':mat[0,1],'primary_tracts':primary},ensure_ascii=False))
