from case_config import CASE
from pathlib import Path
import json,html,base64
import pandas as pd
R = CASE.root
O=R/'analysis_glm_regions'
summary=json.loads((O/'summary.json').read_text(encoding='utf-8'))
peaks=pd.read_csv(O/'tables/all_peak_labels.tsv',sep='\t').fillna('').to_dict('records')
overlaps=pd.read_csv(O/'tables/all_cluster_overlap.tsv',sep='\t').fillna('').to_dict('records')
totals=pd.read_csv(O/'tables/all_region_totals.tsv',sep='\t').fillna('').to_dict('records')
def e(v):return html.escape(str(v))
def img(name):return '<img src="data:image/png;base64,'+base64.b64encode((O/'qc'/name).read_bytes()).decode()+'" alt="'+e(name)+'">'
def download(rel,label):
 p=O/rel;return '<a download="'+e(p.name)+'" href="data:text/tab-separated-values;charset=utf-8;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'+e(label)+'</a>'
def coord(p):return ', '.join(f'{p[x]:.1f}' for x in ['peak_x_mm','peak_y_mm','peak_z_mm'])
body='<header><h1>GLM脑区标注</h1><p>Harvard–Oxford皮层及皮层下图谱 · MNI2009标准空间</p><nav><a href="#strongest">最强峰值</a><a href="#explore">峰值与整簇重叠</a><a href="#regional">脑区汇总</a><a href="#method">方法</a></nav></header>'
body+='<section id="strongest"><h2>各结果中反应最强的体素</h2><p>任务高于静息取最大正Z；静息高于任务取最小负Z。脑区名称查询该体素本身。</p><div class="scroll"><table><thead><tr><th>结果</th><th>方向</th><th>Z</th><th>MNI坐标（mm）</th><th>皮层图谱标签</th><th>皮层下图谱标签</th></tr></thead><tbody>'
run_cn={'run-'+r:'run '+r for r in CASE.runs}|{'combined':'合并'}
for p in summary['strongest_peaks']:
 body+='<tr>'+''.join('<td>'+e(x)+'</td>' for x in [run_cn[p['run']],'任务＞静息' if p['direction']=='task>rest' else '静息＞任务',f'{p["peak_z"]:.3f}',coord(p),p['cortical_peak_region_cn'],p['subcortical_peak_region_cn'],p['aal3_peak_region_cn']])+'</tr>'
body+='</tbody></table></div><details><summary>查看峰值切片</summary><div class="gallery">'
for p in summary['strongest_peaks']:
 tag='positive' if p['direction']=='task>rest' else 'negative';body+=img(f'peak_{p["run"]}_{tag}.png')
body+='</div></details></section>'
body+='''<section id="composition"><h2>脑区占比</h2><div class="controls"><label>结果 <select id="chart-run"><option value="combined">合并</option>__RUN_OPTIONS__</select></label><label>方向 <select id="chart-direction"><option value="task>rest">任务高于静息</option><option value="rest>task">静息高于任务</option></select></label><label>簇 <select id="chart-cluster"></select></label><label>图谱 <select id="chart-atlas"><option value="cortical">皮层图谱</option><option value="subcortical">皮层下图谱</option></select></label></div><p id="chart-caption"></p><div id="cluster-composition"></div></section>'''
body+='''<section id="explore"><h2>峰值与整簇脑区重叠</h2><div class="controls"><label>结果 <select id="run"><option value="combined">合并</option>__RUN_OPTIONS__</select></label><label>方向 <select id="direction"><option value="task>rest">任务高于静息</option><option value="rest>task">静息高于任务</option></select></label><label>排序 <select id="sort"><option value="z">峰值强度 |Z|</option><option value="size">簇体积</option></select></label><label>最少体素数 <input id="minimum" type="number" min="1" value="1"></label></div><p id="count"></p><p class="note">点击“脑区比例”查看完整重叠表。百分比以整个簇为分母；皮层、皮层下图谱各自计算，不相加。</p><div class="scroll"><table id="peaks"><thead><tr><th>簇ID</th><th>体素数</th><th>体积（mm³）</th><th>峰值Z</th><th>MNI坐标</th><th>峰值皮层标签</th><th>峰值皮层下标签</th><th>整簇</th></tr></thead><tbody></tbody></table></div><p id="empty" hidden>当前条件下没有簇。</p></section>'''
body+='''<section id="regional"><h2>所选结果的脑区汇总</h2><p>合并同一方向的所有显著体素，统计各脑区内的数量；这反映分布范围，不是反应强度排名；上方的最少体素数仅筛选簇表，不影响此汇总。</p><label>图谱 <select id="atlas"><option value="cortical">皮层图谱</option><option value="subcortical">皮层下图谱</option></select></label><div class="scroll"><table id="regions"><thead><tr><th>脑区</th><th>显著体素数</th><th>占该方向全部显著体素</th><th>脑区内峰值Z</th></tr></thead><tbody></tbody></table></div></section>'''
body+='<section id="method"><h2>方法与结果范围</h2><ul><li>沿用现有双侧体素水平FDR q&lt;0.05结果，没有重跑GLM或改变阈值。簇是6邻域连通分组，没有增加最小簇大小，也没有作簇水平显著性检验。</li><li>每簇保留一个最强峰值；没有额外拆分所有局部峰值。列表按|Z|排序，体积小的簇也保留。</li><li>使用最大概率、25%阈值的Harvard–Oxford图谱。原MNI6图谱经已有MNI6→患者T1→MNI2009变换，以一次最近邻插值映射至GLM网格。</li><li>皮层图谱在源空间区分左右。皮层下图谱副本采用aseg编号，已按原XML顺序核对标签。</li><li>“某脑区占簇的比例”＝该脑区标签内的簇体素数÷整个簇体素数。它不是激活概率，也不是原概率图谱的成员概率。</li><li>“未分配”表示图谱未覆盖、25%阈值留白或边界位置，未用最近脑区代填。Harvard–Oxford不提供详细小脑分区。</li><li>图谱标注是解剖位置近似。中央前回标签不能单独确认个体手部M1，辅助运动区标签也是图谱参考。</li></ul><details><summary>查看图谱空间对应</summary>'+img('atlas_cortical_MNI2009.png')+img('atlas_subcortical_MNI2009.png')+'</details><p class="links">'
for rel,label in [('tables/all_peak_labels.tsv','全部峰值标注'),('tables/all_cluster_overlap.tsv','全部簇重叠'),('tables/all_region_totals.tsv','脑区汇总数据'),('tables/strongest_peak_per_direction.tsv','各方向最强峰值'),('tables/atlas_coverage_comparison.tsv','图谱覆盖对照')]:body+=download(rel,label)
body+='</p><p class="note">图谱与查询方法：<a href="https://fsl.fmrib.ox.ac.uk/fsl/docs/other/datasets.html">FSL图谱说明</a> · <a href="https://fsl.fmrib.ox.ac.uk/fsl/docs/utilities/dataset_clitools.html">图谱查询说明</a></p></section>'
body=body.replace('__RUN_OPTIONS__', ''.join('<option value="run-'+r+'">run '+r+'</option>' for r in CASE.runs))
script=r'''
const DATA=__DATA__;
const $=id=>document.getElementById(id),td=(tr,v)=>{let x=document.createElement('td');x.textContent=v;tr.appendChild(x);return x;};
function overlapTable(run,cid,atlas){
 const data=DATA.overlaps.filter(x=>x.run===run&&x.cluster_id===cid&&x.atlas===atlas).sort((a,b)=>b.voxels-a.voxels);
 const box=document.createElement('div'),h=document.createElement('h4');h.textContent=ATLAS_NAMES[atlas];box.appendChild(h);
 box.appendChild(regionComposition(data,DATA.peaks.find(x=>x.run===run&&x.cluster_id===cid).voxels,atlas));
 const t=document.createElement('table'),head=t.createTHead().insertRow();['脑区','体素数','占整个簇'].forEach(s=>{let th=document.createElement('th');th.textContent=s;head.appendChild(th)});
 const b=t.createTBody();data.forEach(x=>{const tr=b.insertRow();td(tr,x.region_cn);td(tr,x.voxels);td(tr,x.percent_of_entire_cluster.toFixed(1)+'%');});box.appendChild(t);return box;
}
function render(){
 const run=$('run').value,direction=$('direction').value,min=Math.max(1,Number($('minimum').value)||1);
 const group=DATA.peaks.filter(x=>x.run===run&&x.direction===direction),data=group.filter(x=>x.voxels>=min);
 data.sort($('sort').value==='size'?(a,b)=>b.voxels-a.voxels:(a,b)=>Math.abs(b.peak_z)-Math.abs(a.peak_z));
 const count=group.reduce((s,x)=>s+x.voxels,0);$('count').textContent=`该方向共 ${count} 个显著体素、${group.length} 个连通簇；当前显示 ${data.length} 个簇。`;
 const b=$('peaks').tBodies[0];b.replaceChildren();$('empty').hidden=!!data.length;
 data.forEach(x=>{
  let tr=b.insertRow();[x.cluster_id,x.voxels,x.volume_mm3.toFixed(0),x.peak_z.toFixed(3),[x.peak_x_mm,x.peak_y_mm,x.peak_z_mm].map(y=>y.toFixed(1)).join(', '),x.cortical_peak_region_cn,x.subcortical_peak_region_cn,x.aal3_peak_region_cn].forEach(v=>td(tr,v));
  const cell=td(tr,''),button=document.createElement('button');button.textContent='脑区比例';button.setAttribute('aria-expanded','false');cell.appendChild(button);
  let detail=b.insertRow();detail.hidden=true;detail.className='detail';let d=detail.insertCell();d.colSpan=9;const wrap=document.createElement('div');wrap.className='overlap';d.appendChild(wrap);
  button.onclick=()=>{detail.hidden=!detail.hidden;button.setAttribute('aria-expanded',String(!detail.hidden));if(!detail.hidden&&!wrap.childElementCount){wrap.appendChild(overlapTable(run,x.cluster_id,'cortical'));wrap.appendChild(overlapTable(run,x.cluster_id,'subcortical'));wrap.appendChild(overlapTable(run,x.cluster_id,'aal3'));}};
 });
 const atlas=$('atlas').value,r=$('regions').tBodies[0];r.replaceChildren();DATA.totals.filter(x=>x.run===run&&x.direction===direction&&x.atlas===atlas).sort((a,b)=>b.significant_voxels-a.significant_voxels).forEach(x=>{let tr=r.insertRow();[x.region_cn,x.significant_voxels,x.percent_of_direction_voxels.toFixed(1)+'%',x.regional_peak_z.toFixed(3)].forEach(v=>td(tr,v));});
 renderRegionCharts();
}
['run','direction','sort','minimum','atlas'].forEach(id=>$(id).addEventListener('change',render));bindRegionCharts();render();
'''.replace('__DATA__',json.dumps({'peaks':peaks,'overlaps':overlaps,'totals':totals},ensure_ascii=False,allow_nan=False).replace('</',r'<\/'))
body=body.replace('点击“脑区比例”查看完整重叠表。','点击“脑区比例”查看占比图和完整重叠表。')
body=body.replace('Harvard–Oxford皮层及皮层下图谱 · MNI2009标准空间','Harvard–Oxford＋AAL3 · MNI2009标准空间')
body=body.replace('图谱与查询方法：','图谱与查询方法：<a href="https://www.oxcns.org/aal3.html">AAL3作者说明</a> · ')
body=body.replace('<th>皮层下图谱标签</th>','<th>HO皮层下标签</th><th>AAL3标签</th>').replace('<th>皮层图谱标签</th>','<th>HO皮层标签</th>')
body=body.replace('<th>峰值皮层下标签</th>','<th>峰值HO皮层下标签</th><th>峰值AAL3标签</th>').replace('<th>峰值皮层标签</th>','<th>峰值HO皮层标签</th>')
body=body.replace('<option value="cortical">皮层图谱</option>','<option value="cortical">HO皮层图谱</option>').replace('<option value="subcortical">皮层下图谱</option>','<option value="subcortical">HO皮层下图谱</option><option value="aal3" selected>AAL3图谱</option>')
body=body.replace('皮层、皮层下图谱各自计算，不相加。','三套图谱各自计算，不相加。')
body=body.replace('<details><summary>查看图谱空间对应</summary>','<p>AAL3采用2024年v2发布包，其中标签影像仍名为AAL3v1，共166个分区。按Colin27→患者T1→MNI2009完成空间对应；AAL3是确定性标签图，不使用HO的25%概率阈值。微小核团在本次2 mm网格上仅作位置参考。</p><details><summary>查看图谱空间对应</summary>'+img('atlas_AAL3_MNI2009.png')+img('AAL3_registration_edges.png'))
coverage=pd.read_csv(O/'tables/atlas_coverage_comparison.tsv',sep='\t')
cover='<details><summary>查看未分配比例对照</summary><table><thead><tr><th>结果</th><th>方向</th><th>HO皮层</th><th>HO皮层下</th><th>AAL3</th></tr></thead><tbody>'
for run in ['run-'+r for r in CASE.runs]+['combined']:
 for direction in ['task>rest','rest>task']:
  g=coverage[(coverage.run==run)&(coverage.direction==direction)].set_index('atlas');cover+='<tr>'+''.join('<td>'+e(x)+'</td>' for x in [run_cn[run],'任务＞静息' if direction=='task>rest' else '静息＞任务',*[f'{g.loc[k,"unassigned_percent"]:.1f}%' for k in ['cortical','subcortical','aal3']]])+'</tr>'
cover+='</tbody></table><p class="note">分母为该结果、该方向的全部显著体素；图谱的覆盖范围与分区边界不同。</p></details>'
body=body.replace('<details><summary>查看峰值切片</summary>',cover+'<details><summary>查看峰值切片</summary>')
body=body.replace('<a href="#explore">峰值与整簇重叠</a>','<a href="#composition">脑区占比</a><a href="#explore">峰值与整簇重叠</a>')
body=body.replace('<div class="scroll"><table id="regions">','<p class="note">图中显示前12个已命名脑区，“未分配”单列，其余合并为“其他脑区”；下表保留全部结果。</p><div id="regional-chart"></div><div class="scroll"><table id="regions">')
script=Path(__file__).with_name('glm_region_charts.js').read_text(encoding='utf-8')+'\n'+script
css='body{margin:0;background:#f2f6f9;color:#213746;font:15px/1.8 "Microsoft YaHei",sans-serif}main{max-width:1450px;margin:auto;padding:28px}header p,.note{color:#5d7181}h1{margin:0;font-size:30px}h2{font-size:23px;margin:0 0 10px}nav,.links,.controls{display:flex;gap:20px;flex-wrap:wrap}section{margin:24px 0;background:white;border-radius:12px;padding:24px}a{color:#087b8b}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #dce6ed;vertical-align:top}th{background:#eaf2f6;white-space:nowrap}.scroll{overflow:auto}select,input,button{font:inherit;border:1px solid #bed0dc;border-radius:5px;padding:5px 9px;background:white;color:#214859}input{width:65px}button,summary{cursor:pointer}summary{padding:8px;background:#edf4f7;color:#176c7c}.detail td{background:#f6f9fb}.overlap{display:grid;grid-template-columns:1fr 1fr;gap:24px}h4{margin:10px 0}.gallery img{margin:10px 0}img{width:100%;height:auto}details{margin:18px 0}li{margin:8px 0}@media(max-width:750px){main{padding:12px}section{padding:14px}.overlap{grid-template-columns:1fr}}'
css+='''.composition-chart{margin:14px 0 24px}.composition-strip{display:flex;height:30px;border-radius:5px;overflow:hidden;background:#eef3f6}.composition-strip button{flex-basis:0;min-width:0;padding:0;border:0;border-right:1px solid white;border-radius:0}.composition-strip button:last-child{border-right:0}.composition-strip button:hover,.composition-strip button:focus-visible{filter:brightness(1.2);outline:2px solid #183c50;outline-offset:-2px}.chart-readout{font-size:13px;color:#617486;min-height:24px;margin:6px 0 14px}.chart-axis{display:grid;grid-template-columns:210px 1fr 95px;gap:14px;color:#617486;font-size:12px;margin-bottom:7px}.chart-axis>div{grid-column:2;display:flex;justify-content:space-between}.composition-row{display:grid;grid-template-columns:210px 1fr 95px;align-items:center;gap:14px;min-height:42px;border-bottom:1px solid #edf2f5}.chart-region{font-size:14px;line-height:1.5;padding:6px 0}.chart-track{height:19px;background:repeating-linear-gradient(to right,#eef3f7 0%,#eef3f7 calc(25% - 1px),#cfdae2 calc(25% - 1px),#cfdae2 25%);border-left:1px solid #b4c5d0}.chart-fill{height:100%;min-width:0}.chart-value{text-align:right;white-space:nowrap;line-height:1.3}.chart-value strong{font-size:14px;font-weight:600}.chart-value small{display:block;color:#738797;font-size:11px}.overlap .chart-axis,.overlap .composition-row{grid-template-columns:minmax(115px,40%) 1fr 77px;gap:8px}.overlap .chart-region{font-size:12px}.overlap .chart-axis{font-size:10px}.overlap .chart-value strong{font-size:12px}#chart-caption{font-weight:600;color:#355b70}#chart-cluster{max-width:320px}@media(max-width:750px){.chart-axis,.composition-row{grid-template-columns:minmax(100px,35%) 1fr 70px;gap:7px}.chart-region{font-size:12px}.chart-value strong{font-size:12px}.chart-value small{font-size:10px}.chart-axis{font-size:10px}.controls{gap:10px}.composition-strip{height:25px}.overlap .chart-axis,.overlap .composition-row{grid-template-columns:minmax(100px,35%) 1fr 70px}}'''
css+='.overlap>div:nth-child(3){grid-column:1/-1}.overlap>div:nth-child(3) .chart-axis,.overlap>div:nth-child(3) .composition-row{grid-template-columns:210px 1fr 95px;gap:14px}@media(max-width:750px){.overlap>div:nth-child(3) .chart-axis,.overlap>div:nth-child(3) .composition-row{grid-template-columns:minmax(100px,35%) 1fr 70px;gap:7px}}'
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GLM脑区标注</title><style>'+css+'</style><main>'+body+'</main><script>'+script+'</script></html>'
(O/'GLM脑区标注报告.html').write_text(page,encoding='utf-8')
(O/'结果说明.md').write_text('''# GLM脑区标注

打开同目录“GLM脑区标注报告.html”。结果与原总报告分开。

- tables/all_peak_labels.tsv：每个原有FDR连通簇的一个最强峰值及HO皮层、HO皮层下、AAL3标签。
- tables/all_cluster_overlap.tsv：每簇在三套图谱中的全部重叠，含未分配。
- tables/all_region_totals.tsv：每个run/方向的脑区显著体素汇总。
- tables/strongest_peak_per_direction.tsv：每个run/方向最强的一个峰值。
- tables/atlas_coverage_comparison.tsv：三套图谱的未分配比例对照。
- tables/pre_AAL3/：增加AAL3前的HO标注表备份。
- atlas/：与GLM完全同网格的图谱和中英文字典。
- qc/：图谱与峰值图、验证记录；configuration.json：实际方法和空间变换。

标签比例以整个簇为分母；三套图谱分开统计，不相加。阈值沿用原GLM体素FDR q<0.05，簇不是新增的簇水平检验。未用最近标签代替未分配位置。
AAL3使用官方2024年v2包，标签影像仍为AAL3v1，166个源分区、最大编号170，中间4个原编号留空。源空间Colin27经患者T1映射至GLM MNI2009；详情见atlas/AAL3_registration.json。AAL3不采用HO的25%概率阈值。微小核团在2mm结果上仅作参考。
''',encoding='utf-8')
print(json.dumps({'report_bytes':len(page.encode('utf-8')),'peaks':len(peaks),'overlap_rows':len(overlaps)},ensure_ascii=False))
