"""Whole-trial Morlet ERD using the final report's rest baseline and task windows."""
import argparse
import json
from pathlib import Path
import mne
import numpy as np
from ica_common import TASK_WINDOW, REST_WINDOW, BANDS, write_json, write_csv

SF=250
FREQ=np.arange(6.,36.)
CYCLES=np.full(len(FREQ),5.)
DISPLAY_T=np.arange(-1250,1250,5)/SF
PARAMETERS=dict(frequencies_hz=FREQ.tolist(),n_cycles=5,zero_mean=True,use_fft=True,
    baseline_anchor='preceding_rest',baseline_s=list(REST_WINDOW),task_s=list(TASK_WINDOW),
    display_s=[-5,5],display_step_s=.02,ptp_reject_volts=150e-6,
    acceptance='complete display AND complete independent rest baseline; no partial trials',
    bands={k:list(v) for k,v in BANDS.items()},band_endpoints='inclusive')


def quality(data,bad,start,stop):
    if start<0 or stop>data.shape[1]:return ['out_of_record']
    x=data[:,start:stop];reasons=[]
    if bad[start:stop].any():reasons.append('bad_or_local_source')
    if not np.isfinite(x).all():reasons.append('nonfinite')
    if np.ptp(x,axis=1).max()>=150e-6:reasons.append('ptp_ge_150uV')
    return reasons


def prepare(raw,psd_dir,out,bad):
    trials=json.loads((Path(psd_dir)/'trials.json').read_text(encoding='utf-8'))
    data=raw.get_data()
    kernels=mne.time_frequency.morlet(SF,FREQ,n_cycles=CYCLES,zero_mean=True)
    radius=max((len(k)-1)//2 for k in kernels)
    audit=[];displays=[];base_freq=[];base_band=[];task_band=[];effects=[];onsets=[];ids=[];clusters=[];baselines=[];rows=[]
    for trial in trials:
        onset=round(trial['mi_onset_s']*SF);ds=onset-1250;de=onset+1250
        bs=trial['rest_start_sample'];be=trial['rest_stop_sample']
        ts=trial['task_start_sample'];te=trial['task_stop_sample']
        reasons=['display_'+r for r in quality(data,bad,ds,de)]
        if bs is None:reasons.append('missing_preceding_rest')
        else:
            reasons+=['baseline_'+r for r in quality(data,bad,bs,be)]
            if be>trial['rest_state_stop_sample']:reasons.append('baseline_crosses_next_state')
        if te>trial['task_state_stop_sample']:reasons.append('task_crosses_next_state')
        start=min(ds,bs if bs is not None else ds,ts)-radius
        stop=max(de,be if be is not None else de,te)+radius
        if start<0 or stop>raw.n_times:reasons.append('wavelet_padding_out_of_record')
        elif not np.isfinite(data[:,start:stop]).all():reasons.append('nonfinite_extraction')
        audit.append(dict(trial_id=trial['trial_id'],cluster=trial['cluster'],mi_onset_s=trial['mi_onset_s'],
            rest_onset_s=trial['rest_onset_s'],baseline_start_s=bs/SF if bs is not None else None,
            baseline_stop_s=be/SF if be is not None else None,accepted=not reasons,reasons=';'.join(reasons)))
        if reasons:continue
        power=mne.time_frequency.tfr_array_morlet(data[None,:,start:stop],SF,FREQ,n_cycles=CYCLES,
            zero_mean=True,use_fft=True,decim=1,output='power',n_jobs=1,verbose=False)[0]
        baseline=power[:,:,bs-start:be-start];task=power[:,:,ts-start:te-start]
        bp=np.stack([baseline[:,(FREQ>=lo)&(FREQ<=hi)].mean(axis=(1,2)) for lo,hi in BANDS.values()],axis=-1)
        tp=np.stack([task[:,(FREQ>=lo)&(FREQ<=hi)].mean(axis=(1,2)) for lo,hi in BANDS.values()],axis=-1)
        effect=10*np.log10(tp/bp)
        displays.append(power[:,:,np.arange(ds-start,de-start,5)])
        base_freq.append(baseline.mean(axis=-1));base_band.append(bp);task_band.append(tp);effects.append(effect)
        onsets.append(trial['mi_onset_s']);ids.append(trial['trial_id']);clusters.append(trial['cluster'])
        baselines.append([(bs-onset)/SF,(be-onset)/SF])
        for j,ch in enumerate(raw.ch_names):
            for k,band in enumerate(BANDS):
                rows.append(dict(trial_id=trial['trial_id'],cluster=trial['cluster'],channel=ch,band=band,
                    baseline_power=float(bp[j,k]),task_power=float(tp[j,k]),effect_db=float(effect[j,k])))
    out=Path(out);out.mkdir(parents=True,exist_ok=False);nch=len(raw.ch_names)
    stack=lambda values,shape:np.stack(values) if values else np.empty((0,*shape))
    np.savez_compressed(out/'morlet_trials.npz',
        display_power=stack(displays,(nch,len(FREQ),len(DISPLAY_T))),
        baseline_frequency_power=stack(base_freq,(nch,len(FREQ))),
        baseline_band_power=stack(base_band,(nch,len(BANDS))),task_band_power=stack(task_band,(nch,len(BANDS))),
        effect_db=stack(effects,(nch,len(BANDS))),channels=raw.ch_names,frequencies_hz=FREQ,times_s=DISPLAY_T,
        n_cycles=CYCLES,bands=list(BANDS),band_limits_hz=np.array(list(BANDS.values())),trial_ids=ids,
        attempt_clusters=clusters,onsets_s=onsets,baseline_relative_to_mi_s=stack(baselines,(2,)),
        wavelet_half_support_samples=radius,sfreq=SF,power_units='MNE Morlet squared amplitude; not PSD per Hz')
    write_csv(out/'attempt_acceptance.csv',audit,['trial_id','cluster','mi_onset_s','rest_onset_s','baseline_start_s','baseline_stop_s','accepted','reasons'])
    write_csv(out/'per_attempt_bandpower.csv',rows,['trial_id','cluster','channel','band','baseline_power','task_power','effect_db'])
    return dict(status='completed' if ids else 'completed_no_accepted_attempts',parameters=PARAMETERS,
        wavelet_half_support_samples=radius,candidate_attempts=len(trials),accepted_attempts=len(ids),channels=raw.ch_names,
        formulas={'tfr':'10log10(mean_attempt(display_power / per_attempt_frequency_baseline))',
                  'topography':'mean_attempt(10log10(task_band_power / baseline_band_power))'},
        notes=['Whole display and independent baseline must pass QC; every accepted trial has the full 4s task.',
            'Real EEG padding covers the longest wavelet; padding is not an additional artifact rejection window.',
            'Software MI onset is not a confirmed cue/movement onset.',
            'The report does not specify final TFR aggregation order; existing mean-linear-ratio then dB is retained explicitly.',
            'No pixelwise significance tests; display uses 20ms samples, all power aggregation uses 250Hz.'])


def load_results(directory):
    directory=Path(directory)
    return np.load(directory/'morlet_trials.npz',allow_pickle=False),json.loads((directory/'morlet_manifest.json').read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clean',required=True,type=Path);p.add_argument('--psd',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    raw=mne.io.read_raw_fif(a.clean/'clean_raw.fif',preload=True)
    mask=np.load(a.clean/'local_source_mask.npz')['bad_mask']
    report=prepare(raw,a.psd,a.output,mask)
    report.update(source_clean=str(a.clean.resolve()),source_psd=str(a.psd.resolve()),
        versions={'mne':mne.__version__,'numpy':np.__version__})
    write_json(a.output/'morlet_manifest.json',report);print(report['status'],report['accepted_attempts'])


if __name__=='__main__':main()
