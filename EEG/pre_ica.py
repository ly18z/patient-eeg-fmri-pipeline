"""Fixed EEG preprocessing: select channels, notch, bandpass, resample, reference."""
import argparse
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np
import scipy

PARAMETERS = {
    "notch": {"freqs": [50.0], "trans_bandwidth": 3.0,
              "filter_length": "auto", "notch_widths": None, "method": "fir",
              "phase": "zero", "fir_window": "hamming", "fir_design": "firwin",
              "pad": "reflect_limited"},
    "bandpass": {"l_freq": 1.0, "h_freq": 45.0, "method": "iir",
                 "iir_params": {"order": 4, "ftype": "butter", "output": "sos"},
                 "phase": "zero", "pad": "reflect_limited"},
    "resample": {"sfreq": 250.0, "method": "fft", "npad": "auto",
                 "window": "boxcar", "pad": "reflect_limited"},
    "skip_by_annotation": ["edge", "bad_acq_skip"],
    "reference": "instantaneous mean of selected effective EEG channels; applied",
    "rank": {"tol": 1e-6, "tol_kind": "relative", "proj": False},
}


def resolve_channels(raw):
    """Pick EEG channels from the input header after explicit auxiliary/bad/flat exclusions."""
    auxiliary_names = {"a1", "a2", "m1", "m2", "ref", "reference", "gnd", "ground",
                       "status", "trigger", "trig", "marker", "annotations", "bdfannotations"}
    excluded = []
    candidates = []
    for name, kind in zip(raw.ch_names, raw.get_channel_types()):
        normalized = re.sub(r"[\s_.]", "", name).lower()
        normalized = re.sub(r"^eeg", "", normalized)
        reason = None
        if kind != "eeg":
            reason = "header_type_" + kind
        elif name in raw.info["bads"]:
            reason = "marked_bad_in_input"
        elif normalized in auxiliary_names or re.fullmatch(r"(?:v?eog|heog|ecg|ekg|emg)\d*", normalized):
            reason = "recognized_auxiliary_or_reference_name"
        if reason:
            excluded.append({"channel": name, "reason": reason})
        else:
            candidates.append(name)
    for name, values in zip(candidates, raw.get_data(picks=candidates)):
        if np.ptp(values) == 0:
            excluded.append({"channel": name, "reason": "whole_record_exact_flatline"})
    flat = {item["channel"] for item in excluded}
    candidates = [name for name in candidates if name not in flat]
    return candidates, {"mode": "auto_header", "excluded": excluded}


def preprocess(raw):
    """Return ICA-ready EEG only. Annotation labels are preserved, never remapped."""
    channels, selection = resolve_channels(raw)
    raw = raw.copy().pick(channels).load_data()
    raw.notch_filter(**PARAMETERS["notch"], picks="all", n_jobs=1,
                     skip_by_annotation=PARAMETERS["skip_by_annotation"])
    raw.filter(**PARAMETERS["bandpass"], picks="all", n_jobs=1,
               skip_by_annotation=PARAMETERS["skip_by_annotation"])
    raw.resample(**PARAMETERS["resample"], n_jobs=1)
    raw.set_eeg_reference(ref_channels="average", projection=False, ch_type="eeg")
    rank = mne.compute_rank(raw, **PARAMETERS["rank"])["eeg"]
    return raw, {"selected_channels": raw.ch_names, "channel_selection": selection,
                 "measured_eeg_rank": int(rank)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="BDF/EDF/FIF source")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    args = parser.parse_args()
    source = args.input.resolve()
    args.output.mkdir(parents=True)
    raw = mne.io.read_raw(source, preload=False)
    original_channels = list(raw.ch_names)
    original_types = raw.get_channel_types()
    processed, report = preprocess(raw)
    (args.output / "channel_selection.json").write_text(json.dumps(report["channel_selection"], ensure_ascii=False, indent=2), encoding="utf-8")
    processed.save(args.output / "preica_raw.fif", fmt="double", split_size="2GB", overwrite=False)
    report.update({"status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(source), "parameters": PARAMETERS,
        "original_channels": original_channels, "original_channel_types": original_types,
        "excluded_channels": [x for x in original_channels if x not in processed.ch_names],
        "annotation_count": len(processed.annotations),
        "versions": {"python_executable": sys.executable, "python": platform.python_version(),
                     "mne": mne.__version__, "numpy": np.__version__, "scipy": scipy.__version__}})
    (args.output / "run_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "rank": report["measured_eeg_rank"],
                      "channels": len(processed.ch_names)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
