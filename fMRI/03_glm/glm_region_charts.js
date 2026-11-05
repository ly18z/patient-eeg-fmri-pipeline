// Standalone report charts: values come directly from the embedded reviewed TSV rows.
const ATLAS_NAMES={cortical:'HO皮层图谱',subcortical:'HO皮层下图谱',aal3:'AAL3图谱'};
function regionColor(atlas,id){
 if(id===0)return '#9aa5b1';if(id===-1)return '#c0c9d1';
 const family=atlas==='cortical'?Math.floor((id-1)/2):id;
 const hue=(family*137.508+198)%360,light=atlas==='cortical'&&id%2===0?58:46;
 return `hsl(${hue.toFixed(1)} 51% ${light}%)`;
}
function regionComposition(rows,denominator,atlas,{limit=false}={}){
 let data=rows.map(x=>({id:x.label_id,name:x.region_cn,voxels:x.voxels??x.significant_voxels})).sort((a,b)=>b.voxels-a.voxels);
 if(limit){
  const named=data.filter(x=>x.id!==0),unassigned=data.filter(x=>x.id===0),tail=named.slice(12);
  data=named.slice(0,12).concat(unassigned);
  if(tail.length)data.push({id:-1,name:'其他脑区',voxels:tail.reduce((s,x)=>s+x.voxels,0)});
  data.sort((a,b)=>b.voxels-a.voxels);
 }
 const box=document.createElement('div');box.className='composition-chart';box.dataset.denominator=String(denominator);box.dataset.atlas=atlas;
 const strip=document.createElement('div');strip.className='composition-strip';strip.setAttribute('aria-label','脑区占比，合计100%');box.appendChild(strip);
 const readout=document.createElement('p');readout.className='chart-readout';readout.textContent=`分母：${denominator.toLocaleString()} 个体素 · 合计100%`;
 const reset=()=>readout.textContent=`分母：${denominator.toLocaleString()} 个体素 · 合计100%`;
 data.forEach(x=>{
  const percent=100*x.voxels/denominator,b=document.createElement('button');
  b.type='button';b.style.flexGrow=String(x.voxels);b.style.background=regionColor(atlas,x.id);b.dataset.percent=String(percent);b.dataset.voxels=String(x.voxels);b.dataset.labelId=String(x.id);
  b.setAttribute('aria-label',`${x.name}：${percent.toFixed(1)}%，${x.voxels}个体素`);
  const show=()=>readout.textContent=`${x.name}：${percent.toFixed(1)}%（${x.voxels.toLocaleString()} 个体素）`;
  b.addEventListener('pointerenter',show);b.addEventListener('pointerleave',reset);b.addEventListener('focus',show);b.addEventListener('blur',reset);strip.appendChild(b);
 });
 box.appendChild(readout);
 const axis=document.createElement('div');axis.className='chart-axis';const ticks=document.createElement('div');[0,25,50,75,100].forEach(n=>{const t=document.createElement('span');t.textContent=n+'%';ticks.appendChild(t)});axis.appendChild(ticks);box.appendChild(axis);
 const list=document.createElement('div');list.className='composition-rows';box.appendChild(list);
 data.forEach(x=>{
  const percent=100*x.voxels/denominator,row=document.createElement('div');row.className='composition-row';row.dataset.percent=String(percent);row.dataset.voxels=String(x.voxels);row.dataset.labelId=String(x.id);
  const label=document.createElement('span');label.className='chart-region';label.textContent=x.name.replace('（Juxtapositional Lobule）','');label.title=x.name;
  const track=document.createElement('div');track.className='chart-track';const fill=document.createElement('div');fill.className='chart-fill';fill.style.width=percent+'%';fill.style.background=regionColor(atlas,x.id);track.appendChild(fill);
  const value=document.createElement('span');value.className='chart-value';const pct=document.createElement('strong');pct.textContent=percent.toFixed(1)+'%';value.appendChild(pct);const n=document.createElement('small');n.textContent=x.voxels.toLocaleString()+' 体素';value.appendChild(n);
  row.append(label,track,value);list.appendChild(row);
 });
 return box;
}
let chartGroup='';
function renderRegionCharts(){
 const run=$('run').value,direction=$('direction').value,atlas=$('chart-atlas').value;
 $('chart-run').value=run;$('chart-direction').value=direction;
 const group=DATA.peaks.filter(x=>x.run===run&&x.direction===direction).sort((a,b)=>b.voxels-a.voxels);
 const selector=$('chart-cluster'),groupKey=run+'|'+direction,old=selector.value;selector.replaceChildren();
 group.forEach(x=>{const op=document.createElement('option');op.value=String(x.cluster_id);op.textContent=`簇 ${x.cluster_id} · ${x.voxels.toLocaleString()} 体素 · Z ${x.peak_z.toFixed(2)}`;selector.appendChild(op)});
 if(chartGroup===groupKey&&group.some(x=>String(x.cluster_id)===old))selector.value=old;chartGroup=groupKey;
 const chosen=group.find(x=>String(x.cluster_id)===selector.value),target=$('cluster-composition');target.replaceChildren();
 if(chosen){
  const rows=DATA.overlaps.filter(x=>x.run===run&&x.cluster_id===chosen.cluster_id&&x.atlas===atlas);
  target.appendChild(regionComposition(rows,chosen.voxels,atlas));
  $('chart-caption').textContent=`${run==='combined'?'合并结果':run.replace(/^run-/,'run ')} · ${direction==='task>rest'?'任务高于静息':'静息高于任务'} · 簇 ${chosen.cluster_id} · ${ATLAS_NAMES[atlas]}`;
 }else{$('chart-caption').textContent='当前方向没有簇。'}
 const regional=$('regional-chart');regional.replaceChildren();const regionRows=DATA.totals.filter(x=>x.run===run&&x.direction===direction&&x.atlas===$('atlas').value);
 if(regionRows.length)regional.appendChild(regionComposition(regionRows,regionRows.reduce((s,x)=>s+x.significant_voxels,0),$('atlas').value,{limit:true}));
}
function bindRegionCharts(){
 $('chart-run').addEventListener('change',()=>{$('run').value=$('chart-run').value;render();});
 $('chart-direction').addEventListener('change',()=>{$('direction').value=$('chart-direction').value;render();});
 $('chart-atlas').addEventListener('change',renderRegionCharts);
 $('chart-cluster').addEventListener('change',renderRegionCharts);
}
