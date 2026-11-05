"""Shared EEG parameters and numerical utilities."""
import csv
import json
from pathlib import Path
import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

FRONTAL = ['Fp1','Fp2','F7','F3','Fz','F4','F8']
CENTRAL_POSTERIOR = ['C3','Cz','C4','P3','Pz','P4','O1','Oz','O2']
DECISION_POLICY = 'fixed_thresholds_v1'
TASK_WINDOW = (1., 5.)
REST_WINDOW = (6., 10.)
BANDS = {'mu': (8,13), 'beta': (13,30), 'low_beta': (13,20), 'high_beta': (20,30)}
MUSCLE_LOOSE = .15
MUSCLE_STRICT = .512


def suggest_decisions(scores):
    """Literal report thresholds; loose candidates are flags, not extra deletion rules."""
    decisions = []
    for row in scores:
        eye = (row['eog_proxy_max_abs_r'] > .4 and row['frontal_posterior_rms_ratio'] > 3
               and row['low_frequency_fraction'] > .45)
        high_burst = row['high_band_top10_energy_share'] > .5
        muscle = row['muscle_score'] > MUSCLE_STRICT and not high_burst
        loose = row['muscle_score'] > MUSCLE_LOOSE
        if eye or muscle:
            action = 'exclude'
            reason = '；'.join(label for match, label in [
                (eye, '眼电三项规则均满足'), (muscle, '持续肌电联合评分>0.512，Q≤0.5')] if match)
        elif high_burst:
            action, reason = 'local', '高频异常集中于局部时段，按报告保留IC并筛选污染片段；可人工覆盖'
        else:
            action = 'keep'
            reason = '宽松肌电候选（S>0.15），未达自动删除标准，供人工复核' if loose else '未命中全局伪迹规则，继续筛查局部污染'
        decisions.append(dict(ic=row['ic'], action=action, reason=reason, source='automatic',
            review_candidate=loose and action!='exclude'))
    return decisions


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def spans(mask):
    d = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def valid_samples(raw):
    return np.isfinite(raw.get_data(reject_by_annotation='NaN')).all(axis=0)


def segmented_bandpass(data, good, sf, low, high):
    out = np.full_like(data, np.nan)
    sos = butter(4, [low, high], btype='bandpass', fs=sf, output='sos')
    for a, b in spans(good):
        if b-a >= int(sf):
            out[:, a:b] = sosfiltfilt(sos, data[:, a:b], axis=-1)
    return out


def source_features(source, good, sf):
    """Continuous-source filtering then anchored 0.5s blocks; never concatenate gaps."""
    block = int(round(sf*.5))
    n = source.shape[1]//block
    z = np.full((len(source), 2, n), np.nan)
    q = np.full((len(source), 2), np.nan)
    valid = np.zeros((2, n), bool)
    for k, (lo, hi) in enumerate([(1,8),(20,40)]):
        filtered = segmented_bandpass(source, good, sf, lo, hi)
        complete = np.isfinite(filtered[:, :n*block]).all(axis=0).reshape(n, block).all(axis=1)
        valid[k] = complete
        energy = np.mean(filtered[:, :n*block].reshape(len(source), n, block)**2, axis=-1)
        scale = energy[:, complete].mean(axis=1)
        logpower = np.log10(np.maximum(energy/scale[:,None], 1e-12))
        center = np.median(logpower[:,complete], axis=1)
        mad = np.median(abs(logpower[:,complete]-center[:,None]), axis=1)
        z[:,k] = (logpower-center[:,None])/np.maximum(1.4826*mad[:,None], .25)
        top = int(np.ceil(.1*complete.sum()))
        q[:,k] = np.sort(energy[:,complete], axis=1)[:,-top:].sum(axis=1)/energy[:,complete].sum(axis=1)
    return z, q, valid


def source_psd(source, good, sf):
    block = int(round(2*sf))
    starts = [a for lo, hi in spans(good) for a in range(lo, hi-block+1, block)]
    spectra = []
    for a in starts:
        f, p = welch(source[:,a:a+block], fs=sf, window='hann', nperseg=block,
                     noverlap=block//2, nfft=block, detrend='constant', scaling='density')
        spectra.append(p)
    return f, np.mean(spectra, axis=0)


def load_fit_inputs(directory):
    record = json.loads((directory/'fit_manifest.json').read_text(encoding='utf-8'))
    raw = mne.io.read_raw_fif(record['input'], preload=True)
    montage=record['montage']
    raw.set_montage(mne.channels.read_custom_montage(montage) if Path(montage).is_file() else montage)
    ica = mne.preprocessing.read_ica(directory/'decomposition-ica.fif')
    return raw, ica, record
