"""Review report-rule candidates and optionally override automatic IC decisions."""
import argparse
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ica_common import DECISION_POLICY, load_fit_inputs, suggest_decisions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--start',type=float,help='Optional common focus start in seconds')
    p.add_argument('--duration',type=float,default=10)
    a=p.parse_args()
    raw,ica,_=load_fit_inputs(a.fit)
    scores=json.loads((a.fit/'ic_scores.json').read_text(encoding='utf-8'))
    src=ica.get_sources(raw).get_data(); sf=raw.info['sfreq']; t=np.arange(raw.n_times)/sf
    features=np.load(a.fit/'source_features.npz'); f=features['frequencies']; psd=features['psd']
    a.output.mkdir(parents=True); (a.output/'figures').mkdir()
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    for row in scores:
        i=row['ic'];fig=plt.figure(figsize=(12,7),layout='constrained')
        grid=fig.add_gridspec(3,3); top=fig.add_subplot(grid[:2,0]); spec=fig.add_subplot(grid[:2,1]); energy=fig.add_subplot(grid[:2,2])
        ica.plot_components(picks=[i],axes=top,show=False,colorbar=True)
        band=(f>=1)&(f<=45)
        spec.semilogy(f[band],psd[i,band]);spec.set(xlim=(1,45),xlabel='Hz',ylabel='Source PSD (AU squared/Hz)',title='Valid 2s-window Welch PSD')
        bt=(np.arange(features['z'].shape[-1])+.5)*.5
        for k,label in enumerate(['1-8 Hz','20-40 Hz']):energy.plot(bt,features['z'][i,k],label=label,lw=.8)
        energy.axhline(6,c='r',ls='--');energy.axhline(-6,c='r',ls='--');energy.set(xlabel='Seconds from first sample',ylabel='Robust source z',title='0.5s block diagnostic: |z| > 6');energy.legend()
        overview=fig.add_subplot(grid[2,:2]); bins=max(1,int(np.ceil(raw.n_times/1800)))
        n=raw.n_times//bins; s=src[i,:n*bins].reshape(n,bins); tt=t[:n*bins].reshape(n,bins).mean(axis=1)
        overview.fill_between(tt,s.min(axis=1),s.max(axis=1),color='#345e96',alpha=.7)
        overview.set(xlabel='Seconds from first sample',ylabel='Source (arbitrary units)',title='Full-record min/max envelope (display only)')
        focus=fig.add_subplot(grid[2,2]); duration=min(a.duration,raw.n_times/sf)
        start=a.start if a.start is not None else max(0,min(float(np.argmax(abs(src[i]))/sf-duration/2),raw.n_times/sf-duration))
        stop=min(raw.n_times/sf,start+duration); lo=int(start*sf); hi=int(stop*sf)
        focus.plot(t[lo:hi],src[i,lo:hi],lw=.6)
        focus.set(xlabel='Seconds from first sample',title=f'{start:.2f}-{stop:.2f}s\n'+('Manual interval' if a.start is not None else 'Near peak |source|'))
        fig.suptitle(f'IC {i:02d} (0-based) | muscle S={row["muscle_score"]:.4f} | high-band Q={row["high_band_top10_energy_share"]:.3f}')
        fig.savefig(a.output/f'figures/ic_{i:02d}.png',dpi=125);plt.close(fig)
    (a.output/'index.html').write_text(build_page(scores,a.fit),encoding='utf-8')
    print(a.output/'index.html')


def build_page(scores, fit):
    """Render decisions separately so existing figures can be reused."""
    decisions=suggest_decisions(scores)
    pending_cards=[];automatic_cards=[]
    for row in scores:
        i=row['ic'];decision=decisions[i]
        metrics=html.escape(json.dumps(dict(fit_scores=row,current_rule_decision=decision),ensure_ascii=False,indent=2))
        options=''.join(f'<option value="{value}"'+(' selected' if value==decision['action'] else '')+f'>{label}</option>'
            for value,label in [('keep','保留'),('local','保留，局部污染交给 epoch 筛选'),('exclude','全局删除')])
        label={'keep':'自动保留','local':'自动保留，局部处理','exclude':'自动删除'}[decision['action']]
        card=f'<article data-ic="{i}"><h2>IC {i:02d} · 初始：{label}</h2><img src="figures/ic_{i:02d}.png" alt="IC {i} diagnostics"><details><summary>原拟合指标与当前规则决定</summary><pre>{metrics}</pre></details><label>决定 <select>{options}</select></label> <input class="reason" placeholder="判定依据（可选）" value="{html.escape(decision["reason"],quote=True)}" size="60"></article>'
        (pending_cards if decision['review_candidate'] else automatic_cards).append(card)
    summary=f'<p>宽松候选供复核 {len(pending_cards)} 个；其余规则判定 {len(automatic_cards)} 个；自动全局删除 {sum(d["action"]=="exclude" for d in decisions)} 个。所有决定已按报告阈值填写，人工可覆盖。</p>'
    body=''.join(pending_cards)+f'<details><summary>展开自动判定的 {len(automatic_cards)} 个 IC</summary>'+''.join(automatic_cards)+'</details>'
    payload=json.dumps({'fit':str(Path(fit).resolve()),'component_count':len(scores),'decision_policy':DECISION_POLICY},ensure_ascii=True).replace('<','\\u003c')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ICA 人工审核</title>
<style>body{font:16px system-ui;max-width:1300px;margin:auto;padding:20px;background:#f3f5f8;color:#17283a}article{background:white;padding:18px;margin:20px 0;border-radius:8px}img{width:100%}header{position:sticky;top:0;background:#edf3fa;padding:12px;z-index:2}button,input,select{font:inherit;padding:8px}pre{white-space:pre-wrap}#status{color:#9b2917}</style>
<h1>FastICA 报告阈值与形貌复核</h1><p>编号从0开始，仅对应当前模型。眼电三项同时满足则全局删除，不用Q否决；持续肌电S&gt;0.512且Q≤0.5时全局删除。S&gt;0.15只标记宽松候选，不自动增加删除；高频局部爆发先保留并筛段，其余保留。没有额外“两项眼电指标满足”规则。人工复核只对应本次分解，不能照搬其他记录的IC编号；可在这里覆盖规则决定。此页不改变信号。</p>
<p>空间图正负颜色不代表ERD。Q衡量时间集中性，不能单独证明来源；源波形为任意尺度。额极代理不是真实EOG。末尾不足0.5秒不参与块评分。</p>
<p>局部图默认围绕源绝对峰；需看其他时段可用 show_ica.py --start 秒 --duration 秒另建查看页。完整EEG时间轴始终保留。</p>
<header><label>审核者（可选） <input id="reviewer"></label> <button id="export">导出审核 JSON</button> <label>载入已有名单 <input id="load" type="file" accept=".json"></label><span id="status"></span></header>
''' + summary + body + '<script>const identity='+payload+''';
const cards=[...document.querySelectorAll('article')];
for(const c of cards){const select=c.querySelector('select'),reason=c.querySelector('.reason'),initial={action:select.value,reason:reason.value};select.onchange=()=>{reason.value=select.value===initial.action?initial.reason:''}}
function collect(){return {...identity,reviewer:document.querySelector('#reviewer').value.trim(),decisions:cards.map(c=>({ic:Number(c.dataset.ic),action:c.querySelector('select').value,reason:c.querySelector('.reason').value.trim()}))}}
function restore(d){if(d.fit!==identity.fit)throw Error('拟合目录不匹配');document.querySelector('#reviewer').value=d.reviewer||'';for(const x of d.decisions||[]){let c=cards.find(c=>Number(c.dataset.ic)===x.ic);if(c){c.querySelector('select').value=x.action;c.querySelector('.reason').value=x.reason||''}}}
document.querySelector('#export').onclick=()=>{const d={...collect(),reviewed_at:new Date().toISOString()};const u=URL.createObjectURL(new Blob([JSON.stringify(d,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download='ica_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
document.querySelector('#load').onchange=async e=>{try{restore(JSON.parse(await e.target.files[0].text()));document.querySelector('#status').textContent='已载入'}catch(e){document.querySelector('#status').textContent=String(e)}};
</script></html>'''
    return page


if __name__=='__main__':main()
