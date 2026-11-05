"""Apply automatic IC decisions with optional manual overrides, then mark local contamination."""
import argparse
import csv
import json
from pathlib import Path
import mne
import numpy as np
from ica_common import DECISION_POLICY, load_fit_inputs, suggest_decisions, valid_samples, write_json, spans


def resolve_decisions(scores, review=None):
    decisions={row['ic']:row for row in suggest_decisions(scores)}
    for row in review['decisions'] if review else []:
        current=decisions[row['ic']]
        reason=row.get('reason','')
        if row['action']!=current['action'] or reason!=current['reason']:
            decisions[row['ic']]=dict(ic=row['ic'],action=row['action'],reason=reason,source='manual')
    if all(row['action']=='exclude' for row in decisions.values()):raise ValueError('Cannot exclude every component')
    return list(decisions.values())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',required=True,type=Path);p.add_argument('--review',type=Path,help='Optional morphology overrides; omit for the report threshold-only branch')
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    raw,ica,_=load_fit_inputs(a.fit)
    scores=json.loads((a.fit/'ic_scores.json').read_text(encoding='utf-8'))
    review=json.loads(a.review.read_text(encoding='utf-8-sig')) if a.review else None
    decisions=resolve_decisions(scores,review)
    excluded=sorted(row['ic'] for row in decisions if row['action']=='exclude')
    good=valid_samples(raw)
    with np.load(a.fit/'source_features.npz') as features:
        z=features['z'];valid=features['valid_blocks']
    kept=[i for i in range(ica.n_components_) if i not in excluded]
    block=125;n=z.shape[-1]; valid_block=valid.all(axis=0)
    dirty=(np.abs(z[kept])>6).any(axis=(0,1)) | ~valid_block
    mask=np.ones(raw.n_times,bool)  # partial final block remains unusable for local screening
    mask[:n*block]=np.repeat(dirty,block);mask|=~good
    ica.exclude=excluded
    clean=ica.apply(raw.copy())
    clean.set_eeg_reference('average',projection=False)
    for lo,hi in spans(mask):
        clean.annotations.append(raw.first_time+lo/250,(hi-lo)/250,'BAD_ICA_local_source')
    a.output.mkdir(parents=True)
    clean.save(a.output/'clean_raw.fif',fmt='double',overwrite=False)
    ica.save(a.output/'applied-ica.fif',overwrite=False)
    with (a.output/'local_source_blocks.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['block','start_s','stop_s','valid_reference_block','source_contaminated'])
        for i in range(n):w.writerow([i,i*.5,(i+1)*.5,bool(valid_block[i]),bool(dirty[i])])
    np.savez_compressed(a.output/'local_source_mask.npz',bad_mask=mask,
                        retained_ics=kept,valid_blocks=valid,bands=['1-8','20-40'],sfreq=250)
    write_json(a.output/'apply_manifest.json',dict(status='cleaned_local_mask_ready_no_epochs',
         decision_policy=DECISION_POLICY,review=review,decisions=decisions,actual_excluded=excluded,retained=kept,
         selection_mode='report thresholds with morphology overrides' if review else 'report thresholds only',
         automatically_excluded=[row['ic'] for row in decisions if row['action']=='exclude' and row['source']=='automatic'],
         manually_excluded=[row['ic'] for row in decisions if row['action']=='exclude' and row['source']=='manual'],
         samples= int(raw.n_times),sfreq=250,original_annotations=len(raw.annotations),
         local_source_rule='any retained IC / band abs(z)>6 in a 0.5s block',
         local_bad_seconds=float(mask.sum()/250),n_components_removed=len(excluded),
         amplitude_rule_for_future_epochs='any effective EEG PTP >=150e-6 V rejects epoch; not yet executed',
         notes=['Time axis retained; appended BAD_ICA_local_source; no epochs rejected here.',
                'Invalid blocks and unscored partial final blocks are marked unusable.',
                'All retained ICs screened; not only muscle candidates.'],
         versions={'mne':mne.__version__,'numpy':np.__version__}))
    print('Applied exclusions:',excluded)


if __name__=='__main__':main()
