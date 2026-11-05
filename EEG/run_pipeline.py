"""Run the EEG pipeline from its own configuration."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runner_common import entry

HERE = Path(__file__).resolve().parent


def make_plan(config, filename):
    base = filename.parent
    def path(value):
        return (base / value).resolve()
    run = path(config['run_directory'])
    python = str(path(config.get('python', sys.executable)))
    steps = []
    def command(name, argv):
        steps.append(dict(name=name, command=list(map(str, argv))))
    def pause(name, message, pages, **extra):
        steps.append(dict(name=name, message=message, pages=list(map(str, pages)), **extra))

    out = run
    pre, fit, review, clean, psd, morlet, tfr, topo = [out/name for name in
        ['01_preica', '02_ica', '03_ica_review', '04_clean', '05_psd', '06_morlet', '07_tfr', '08_topography']]
    montage = config.get('montage', 'standard_1020')
    if Path(montage).suffix or '/' in montage or '\\' in montage:
        montage = str(path(montage))
    def eeg_step(name, *args):
        command('eeg.'+name, [python, '-X', 'utf8', '-B', HERE/f'{name}.py', *args])
    eeg_step('pre_ica', '--input', path(config['input']), '--output', pre)
    eeg_step('fit_ica', '--input', pre/'preica_raw.fif', '--output', fit, '--montage', montage)
    overrides = []
    if config.get('manual_ica', True):
        eeg_step('show_ica', '--fit', fit, '--output', review)
        review_file = path(config['review_file']) if config.get('review_file') else review/'ica_review.json'
        pause('eeg.review_ica', f'复核候选IC，导出审核JSON并保存到：{review_file}',
              [review/'index.html'], required=str(review_file), fit=str(fit))
        overrides = ['--review', review_file]
    eeg_step('apply_ica', '--fit', fit, *overrides, '--output', clean)
    eeg_step('psd', '--clean', clean, '--output', psd)
    eeg_step('erd', '--clean', clean, '--psd', psd, '--output', morlet)
    channels = ['--channels', *config['tfr_channels']] if config.get('tfr_channels') else []
    eeg_step('plot_tfr', '--morlet', morlet, '--output', tfr, *channels)
    eeg_step('plot_topography', '--morlet', morlet, '--output', topo, '--montage', montage)

    return run, steps


if __name__ == "__main__":
    entry(make_plan, __doc__)
