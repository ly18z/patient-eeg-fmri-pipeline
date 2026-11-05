"""Paired DPSS PSD/ERD: MI+1–5s versus preceding rest+6–10s."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import false_discovery_control
from ica_common import TASK_WINDOW, REST_WINDOW, BANDS, write_json, write_csv

LENGTHS=(.5,1.,2.)
PERMUTATIONS=100_000
EVENT_CONFIG={
    'event_mapping':{'miSuccess':'attempt','miFailed':'attempt','rest':'rest','initialRest':'rest'},
    'same_state_gap_s':3.0,
    'ignored_annotations':[],
}


def multitaper(epochs, sf):
    """V input; V²/Hz output. NW=2, three low-bias tapers, no zero padding."""
    return mne.time_frequency.psd_array_multitaper(epochs,sfreq=sf,
        bandwidth=4/(epochs.shape[-1]/sf),adaptive=False,low_bias=True,
        normalization='full',remove_dc=True,output='power',n_jobs=1,verbose=False)


def reconstruct_states(raw):
    mapping=EVENT_CONFIG['event_mapping'];gap=EVENT_CONFIG['same_state_gap_s']
    ignored=set(EVENT_CONFIG['ignored_annotations']);events=[];unknown=set()
    for onset,duration,label in zip(raw.annotations.onset,raw.annotations.duration,raw.annotations.description):
        if label not in mapping:
            if not label.lower().startswith(('bad','edge')) and label not in ignored:unknown.add(label)
            continue
        events.append(dict(onset_s=float(onset-raw.first_time),duration_s=float(duration),label=str(label),state=mapping[label]))
    if unknown:raise ValueError('Unmapped annotations: '+str(sorted(unknown)))
    events.sort(key=lambda x:x['onset_s']);groups=[]
    for event in events:
        if groups and event['onset_s']==groups[-1]['last_marker_s'] and event['state']!=groups[-1]['state']:
            raise ValueError('Conflicting simultaneous state markers')
        if not groups or groups[-1]['state']!=event['state'] or event['onset_s']-groups[-1]['last_marker_s']>gap:
            groups.append(dict(cluster=len(groups),state=event['state'],onset_s=event['onset_s'],last_marker_s=event['onset_s'],members=[]))
        groups[-1]['last_marker_s']=event['onset_s'];groups[-1]['members'].append(event)
    for i,g in enumerate(groups):
        g['start_sample']=int(round(g['onset_s']*raw.info['sfreq']))
        g['next_cluster_s']=groups[i+1]['onset_s'] if i+1<len(groups) else raw.n_times/raw.info['sfreq']
    return events,groups


def pair_trials(groups, sf):
    trials=[]
    for i,g in enumerate(groups):
        if g['state']!='attempt':continue
        rest=groups[i-1] if i and groups[i-1]['state']=='rest' else None
        trials.append(dict(trial_id=len(trials),cluster=g['cluster'],rest_cluster=rest['cluster'] if rest else None,
            mi_onset_s=g['onset_s'],rest_onset_s=rest['onset_s'] if rest else None,
            task_start_sample=round((g['onset_s']+TASK_WINDOW[0])*sf),
            task_stop_sample=round((g['onset_s']+TASK_WINDOW[1])*sf),
            rest_start_sample=round((rest['onset_s']+REST_WINDOW[0])*sf) if rest else None,
            rest_stop_sample=round((rest['onset_s']+REST_WINDOW[1])*sf) if rest else None,
            task_state_stop_sample=round(g['next_cluster_s']*sf),
            rest_state_stop_sample=round(rest['next_cluster_s']*sf) if rest else None))
    return trials


def signflip(effects):
    """Two-sided Monte Carlo sign flips; BH-FDR across the supplied feature family."""
    n=effects.shape[0];shape=effects.shape[1:]
    if n<2:return np.full(shape,np.nan),np.full(shape,np.nan)
    values=effects.reshape(n,-1);observed=np.abs(values.sum(axis=0));hits=np.zeros(values.shape[1],int)
    rng=np.random.default_rng(123)
    for start in range(0,PERMUTATIONS,1000):
        signs=rng.choice([-1.,1.],size=(min(1000,PERMUTATIONS-start),n))
        hits+=(np.abs(signs@values)>=observed[None,:]-1e-12).sum(axis=0)
    p=(hits+1)/(PERMUTATIONS+1)
    return p.reshape(shape),false_discovery_control(p,method='bh').reshape(shape)


def plot_spectra(out, T, f, channels, task, rest, mean_db, sd_db, picks, q):
    n=len(task);display=(f>=2)&(f<=40)
    fig,axes=plt.subplots(2,len(picks),figsize=(len(picks)*4.4,7),squeeze=False,layout='constrained')
    for k,j in enumerate(picks):
        top,bottom=axes[:,k]
        for ax in (top,bottom):
            ax.axvspan(8,13,color='#7fbcd1',alpha=.15);ax.axvspan(13,30,color='#d3b483',alpha=.13)
            ax.set(xlim=(2,40),xlabel='Hz');ax.grid(alpha=.15)
        if n:
            top.semilogy(f[display],task[:,j].mean(axis=0)[display]*1e12,label='MI')
            top.semilogy(f[display],rest[:,j].mean(axis=0)[display]*1e12,label='preceding rest',color='gray')
            bottom.plot(f[display],mean_db[j,display],color='#087e98')
            bottom.fill_between(f[display],(mean_db[j]-sd_db[j])[display],(mean_db[j]+sd_db[j])[display],color='#87bdcb',alpha=.35)
        else:bottom.text(.5,.5,'No accepted pairs',transform=bottom.transAxes,ha='center')
        top.set(title=channels[j]+' mean PSD',ylabel='PSD (uV²/Hz)')
        bottom.set(title=f'{channels[j]} paired ERD (n={n})',ylabel='MI/rest power (dB)',ylim=(-8,6))
        bottom.axhline(0,c='black',lw=.8)
        if q is not None:
            for level,threshold in enumerate([.05,.01,.001,.0001]):
                significant=f[display][q[k]<threshold]
                bottom.hlines(np.full(len(significant),-7.8+level*.25),significant-.25,significant+.25,color='#17283a',lw=2)
        if k==0 and n:top.legend()
    suffix=' | sign-flip/BH-FDR within this branch' if q is not None else ''
    fig.suptitle(f'DPSS {T:g}s | NW=2, 3 tapers | MI +1–5s / preceding rest +6–10s\nERD: mean trial dB ± sample SD'+suffix)
    if q is not None:fig.supxlabel('Stacked bars, bottom to top: BH q < 0.05 / 0.01 / 0.001 / 0.0001 | within-record exploratory',fontsize=9)
    fig.savefig(out/f'psd_T{T:g}.png',dpi=150);plt.close(fig)


def compute(raw, mask, out):
    sf=raw.info['sfreq'];data=raw.get_data();bad=mask
    _,groups=reconstruct_states(raw);trials=pair_trials(groups,sf)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'states.json',groups);write_json(out/'trials.json',trials)
    channels=raw.ch_names;nch=len(channels);counts=[]
    picks=[channels.index(c) for c in ['C3','Cz','C4'] if c in channels]
    for T in LENGTHS:
        size=round(T*sf);f=np.fft.rfftfreq(size,1/sf);rows=[];segments=[]
        for trial in trials:
            for condition in ['task','rest']:
                anchor=trial[condition+'_start_sample'];state_stop=trial[condition+'_state_stop_sample']
                for k in range(round(4/T)):
                    start=anchor+k*size if anchor is not None else None;stop=start+size if start is not None else None
                    reasons=[];segment=None;ptp=None
                    if start is None:reasons.append('missing_preceding_rest')
                    elif start<0 or stop>raw.n_times:reasons.append('out_of_record')
                    elif stop>state_stop:reasons.append('crosses_next_state')
                    else:
                        segment=data[:,start:stop];ptp=np.ptp(segment,axis=1)
                        if bad[start:stop].any():reasons.append('bad_or_local_source')
                        if not np.isfinite(segment).all():reasons.append('nonfinite')
                        if ptp.max()>=150e-6:reasons.append('ptp_ge_150uV')
                    rows.append(dict(epoch_id=len(rows),trial_id=trial['trial_id'],cluster=trial['cluster'],condition=condition,
                        T_s=T,start_sample=start,stop_sample=stop,accepted=not reasons,reasons=';'.join(reasons),
                        max_ptp_uV=float(ptp.max()*1e6) if ptp is not None else None))
                    segments.append(segment)
        power=np.full((len(rows),nch,len(f)),np.nan)
        computable=[i for i,x in enumerate(segments) if x is not None and np.isfinite(x).all()]
        for k in range(0,len(computable),64):
            ids=computable[k:k+64];power[ids]=multitaper(np.stack([segments[i] for i in ids]),sf)[0]
        task=[];rest=[];accepted_ids=[];clusters=[];audit=[]
        for trial in trials:
            selections={condition:[r['epoch_id'] for r in rows if r['trial_id']==trial['trial_id'] and r['condition']==condition and r['accepted']] for condition in ['task','rest']}
            accepted=all(selections.values())
            audit.append(dict(trial_id=trial['trial_id'],cluster=trial['cluster'],accepted=accepted,
                task_epochs=len(selections['task']),rest_epochs=len(selections['rest']),
                reason='' if accepted else 'missing_accepted_task_or_rest_epoch'))
            if accepted:
                task.append(power[selections['task']].mean(axis=0));rest.append(power[selections['rest']].mean(axis=0))
                accepted_ids.append(trial['trial_id']);clusters.append(trial['cluster'])
        empty=np.empty((0,nch,len(f)));task=np.stack(task) if task else empty;rest=np.stack(rest) if rest else empty
        db=10*np.log10(task/rest);n=len(task)
        mean_db=db.mean(axis=0) if n else np.full((nch,len(f)),np.nan)
        sd_db=db.std(axis=0,ddof=1) if n>1 else np.full_like(mean_db,np.nan)
        bands=np.stack([10*np.log10(task[:,:,(f>=lo)&(f<=hi)].mean(axis=-1)/rest[:,:,(f>=lo)&(f<=hi)].mean(axis=-1)) for lo,hi in BANDS.values()],axis=-1)
        display=(f>=2)&(f<=40);p=q=band_p=band_q=None
        if T==2:
            p,q=signflip(db[:,picks][:,:,display]);band_p,band_q=signflip(bands[:,picks,:2])
            frequency_rows=[dict(channel=channels[j],frequency_hz=float(freq),n_pairs=n,p=float(p[k,l]) if n>1 else None,q=float(q[k,l]) if n>1 else None)
                for k,j in enumerate(picks) for l,freq in enumerate(f[display])]
            write_csv(out/'frequency_statistics.csv',frequency_rows,['channel','frequency_hz','n_pairs','p','q'])
            band_rows=[dict(channel=channels[j],band=band,n_pairs=n,mean_db=float(bands[:,j,b].mean()) if n else None,
                sd_db=float(bands[:,j,b].std(ddof=1)) if n>1 else None,p=float(band_p[k,b]) if n>1 else None,q=float(band_q[k,b]) if n>1 else None)
                for k,j in enumerate(picks) for b,band in enumerate(['mu','beta'])]
            write_csv(out/'band_summary.csv',band_rows,['channel','band','n_pairs','mean_db','sd_db','p','q'])
        write_csv(out/f'epochs_T{T:g}.csv',rows,['epoch_id','trial_id','cluster','condition','T_s','start_sample','stop_sample','accepted','reasons','max_ptp_uV'])
        write_csv(out/f'paired_trials_T{T:g}.csv',audit,['trial_id','cluster','accepted','task_epochs','rest_epochs','reason'])
        np.savez_compressed(out/f'psd_T{T:g}.npz',psd_v2_hz=power,epoch_accepted=[r['accepted'] for r in rows],
            epoch_trial_ids=[r['trial_id'] for r in rows],epoch_conditions=[r['condition'] for r in rows],
            paired_trial_ids=accepted_ids,attempt_clusters=clusters,task_mean_v2_hz=task,rest_mean_v2_hz=rest,
            trial_erd_db=db,mean_erd_db=mean_db,sd_erd_db=sd_db,band_effect_db=bands,bands=list(BANDS),
            channels=channels,frequencies_hz=f,T_s=T,bandwidth_hz=4/T,sfreq=sf,units='V^2/Hz; ERD in dB')
        plot_spectra(out,T,f,channels,task,rest,mean_db,sd_db,picks,q)
        counts.append(dict(T_s=T,candidate_pairs=len(trials),accepted_pairs=n,
            accepted_task_epochs=sum(r['accepted'] and r['condition']=='task' for r in rows),
            accepted_rest_epochs=sum(r['accepted'] and r['condition']=='rest' for r in rows)))
    return dict(status='completed' if any(c['accepted_pairs'] for c in counts) else 'completed_no_accepted_pairs',
        epoch_lengths_s=list(LENGTHS),parameters=dict(NW=2,n_tapers=3,low_bias=True,adaptive=False,normalization='full',
            remove_dc=True,zero_padding=False,bandwidth='4/T Hz',baseline=None,ptp_reject_v=150e-6,overlap=False,
            task_s=list(TASK_WINDOW),preceding_rest_s=list(REST_WINDOW),display_hz=[2,40]),
        channels=channels,n_samples=int(raw.n_times),sfreq=sf,counts=counts,
        aggregation='average accepted linear PSD within each trial/condition; mean of paired trial dB',
        statistics=dict(primary_epoch_seconds=2,method='two-sided Monte Carlo sign-flip',draws=PERMUTATIONS,seed=123,
            fdr='BH, frequency and mu/beta families separate',channels=[channels[j] for j in picks],
            frequency_tests=len(picks)*77,band_tests=len(picks)*2,scope='current branch only; within-record exploratory'),
        notes=['At least one accepted epoch in EACH condition is required for pairing.',
            'Software markers are alignment proxies, not confirmed cue or movement onsets.',
            'The reference report pooled two branches; its 462/12-test q-values are not copied here.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clean',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    raw=mne.io.read_raw_fif(a.clean/'clean_raw.fif',preload=True)
    mask=np.load(a.clean/'local_source_mask.npz')['bad_mask']
    report=compute(raw,mask,a.output)
    report.update(config=EVENT_CONFIG,source_clean=str(a.clean.resolve()),
        versions={'mne':mne.__version__,'numpy':np.__version__})
    write_json(a.output/'psd_manifest.json',report);print(json.dumps(report['counts'],indent=2))


if __name__=='__main__':main()
