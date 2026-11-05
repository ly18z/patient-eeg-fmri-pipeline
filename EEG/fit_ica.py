"""Fit FastICA and score candidates; never automatically delete components."""
import argparse
import sys
from pathlib import Path
import mne
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits
from ica_common import (FRONTAL, CENTRAL_POSTERIOR, write_json, valid_samples, suggest_decisions,
                        segmented_bandpass, source_features, source_psd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path, help='ICA-ready 250Hz FIF')
    p.add_argument('--output', required=True, type=Path, help='New fit directory')
    p.add_argument('--montage', required=True, help='Confirmed built-in name, e.g. standard_1020, or custom montage file')
    a = p.parse_args()
    raw = mne.io.read_raw_fif(a.input.resolve(), preload=True)
    path = Path(a.montage)
    montage = str(path.resolve()) if path.is_file() else a.montage
    raw.set_montage(mne.channels.read_custom_montage(montage) if path.is_file() else montage)
    good = valid_samples(raw)
    rank = int(mne.compute_rank(raw, tol=1e-6, tol_kind='relative', proj=False)['eeg'])
    params = dict(n_components=rank, method='fastica', random_state=123, max_iter=2000,
                  fit_params={'tol':1e-4, 'algorithm':'parallel', 'fun':'logcosh'})
    ica = mne.preprocessing.ICA(**params)
    with threadpool_limits(limits=1):
        ica.fit(raw, decim=2, reject_by_annotation=True)
    if ica.n_iter_ >= 2000:
        raise RuntimeError('ICA did not converge; no accepted model written')
    src = ica.get_sources(raw).get_data()
    z, q, block_valid = source_features(src, good, 250)
    f, psd = source_psd(src, good, 250)
    _, muscle = ica.find_bads_muscle(raw, threshold=.8, l_freq=7, h_freq=45)
    maps = ica.get_components()
    frontal = [raw.ch_names.index(c) for c in FRONTAL if c in raw.ch_names]
    posterior = [raw.ch_names.index(c) for c in CENTRAL_POSTERIOR if c in raw.ch_names]
    fp1,fp2 = raw.get_data(picks=['Fp1','Fp2'])
    proxy = segmented_bandpass(np.array([(fp1+fp2)/2, fp1-fp2]),good,250,1,10)
    low = segmented_bandpass(src,good,250,1,10)
    valid = good & np.isfinite(low).all(axis=0) & np.isfinite(proxy).all(axis=0)
    corr = [max(abs(np.corrcoef(source[valid],p[valid])[0,1]) for p in proxy) for source in low]
    ratio = np.sqrt(np.mean(maps[frontal]**2,axis=0))/np.sqrt(np.mean(maps[posterior]**2,axis=0))
    integral = lambda lo,hi: np.trapezoid(psd[:,(f>=lo)&(f<=hi)],f[(f>=lo)&(f<=hi)],axis=1)
    low_fraction = integral(1,8)/integral(1,40)
    rows=[]
    for i in range(rank):
        rows.append(dict(ic=i, eog_proxy_max_abs_r=float(corr[i]),
             frontal_posterior_rms_ratio=float(ratio[i]), low_frequency_fraction=float(low_fraction[i]),
             muscle_score=float(muscle[i]), high_band_top10_energy_share=float(q[i,1]),
             low_band_top10_energy_share=float(q[i,0])))
    a.output.mkdir(parents=True)
    ica.save(a.output/'decomposition-ica.fif')
    write_json(a.output/'ic_scores.json',rows)
    np.savez_compressed(a.output/'source_features.npz', z=z,q=q,valid_blocks=block_valid,
                        frequencies=f,psd=psd,block_seconds=.5)
    record=dict(status='fitted_and_scored_no_cleaning', input=str(a.input.resolve()),
         montage=montage,
         channels=raw.ch_names, rank=rank, fit_parameters=params, fit_decim=2,
         fit_sfreq=125, application_sfreq=250, iterations=int(ica.n_iter_),
         valid_samples=int(good.sum()), rejected_samples=int((~good).sum()),
         frontal_channels=[raw.ch_names[i] for i in frontal],
         central_posterior_channels=[raw.ch_names[i] for i in posterior],
         versions={'mne':mne.__version__,'sklearn':sklearn.__version__,'numpy':np.__version__,'python':sys.executable})
    write_json(a.output/'fit_manifest.json',record)
    decisions=suggest_decisions(rows)
    print('Automatic exclusions:',[r['ic'] for r in decisions if r['action']=='exclude'])
    print('Loose candidates for optional morphology review:',[r['ic'] for r in decisions if r['review_candidate']])
    print('Model fitted; apply_ica.py uses report thresholds. show_ica.py supports morphology review and overrides.')


if __name__=='__main__': main()
