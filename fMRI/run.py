"""Run a single MRI stage using portable paths and an installed Docker command."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
IMAGE = 'nipreps/fmriprep:25.2.7'
LOCAL_STEPS = {'prepare_bids', 'audit_dwi_gradients', 'attach_glm_bold_curves',
               'run_chronic_stroke_models', 'run_truenet_wmh'}


def main():
    scripts = {p.stem: p for p in HERE.glob('*/*.py')}
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('step', choices=sorted(scripts) + ['convert', 'preprocess'])
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    os.environ['FMRI_CONFIG'] = str(a.config.resolve())
    from case_config import CASE
    docker = CASE.data.get('docker', 'docker')
    if a.step == 'convert':
        command = [sys.executable, str(HERE/'convert_dicom.py'), '--dicom', str(CASE.dicom),
                   '--output', str(CASE.root/'converted'), '--dcm2niix', CASE.data.get('dcm2niix', 'dcm2niix')]
    elif a.step == 'preprocess':
        c = CASE.data['preprocessing']
        command = [sys.executable, str(HERE/'preprocess.py'), '--bids', str(CASE.root/'bids'),
                   '--participant', CASE.subject.removeprefix('sub-'), '--output',
                   str(CASE.root/'derivatives/fmriprep'), '--templateflow', str(CASE.path(c['templateflow'])),
                   '--license', str(CASE.path(c['license'])), '--docker', docker,
                   '--image', c.get('image', IMAGE)]
        if c.get('ignore'):
            command += ['--ignore', *c['ignore']]
        if c.get('download_templates', False):
            command += ['--allow-template-downloads']
    elif a.step in LOCAL_STEPS:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(HERE) + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        python = sys.executable
        if a.step in {'run_chronic_stroke_models', 'run_truenet_wmh'} and CASE.data.get('lesion_python'):
            python = str((a.config.resolve().parent/CASE.data['lesion_python']).resolve())
        command = [python, '-X', 'utf8', '-B', str(scripts[a.step])]
        if a.dry_run:
            print(json.dumps(command, ensure_ascii=False, indent=2))
            return
        subprocess.run(command, env=env, check=True)
        return
    else:
        if not a.dry_run:
            CASE.tools.mkdir(parents=True, exist_ok=True)
        command = [docker, 'run', '--rm', '--pull', 'never', '--platform', 'linux/amd64', '--network', 'none',
                   '--cpus', '8', '--memory', '13g', '--entrypoint', 'python']
        mounts = [(CASE.root, '/project', False), (HERE, '/pipeline', True),
                  (CASE.tools, '/shared-tools', True), (a.config.resolve(), '/case.json', True)]
        if CASE.font:
            font = '/font/custom'+CASE.font.suffix
            mounts.append((CASE.font, font, True))
            command += ['-e', 'FMRI_FONT='+font]
        for src, dst, readonly in mounts:
            command += ['--mount', f'type=bind,src={src},dst={dst}' + (',readonly' if readonly else '')]
        for value in ['FMRI_CONFIG=/case.json', 'FMRI_ROOT=/project', 'FMRI_TOOLS=/shared-tools',
                      'FMRI_DICOM=/project/dicom', 'PYTHONPATH=/pipeline:/shared-tools/python/dti_linux_packages',
                      'PYTHONDONTWRITEBYTECODE=1', 'MPLCONFIGDIR=/tmp/mpl', 'OMP_NUM_THREADS=4',
                      'OPENBLAS_NUM_THREADS=4', 'LD_LIBRARY_PATH=/app/.pixi/envs/fmriprep/lib']:
            command += ['-e', value]
        command += [CASE.data.get('analysis_image', IMAGE), '-B', '/pipeline/' + scripts[a.step].relative_to(HERE).as_posix()]
    if a.dry_run:
        print(json.dumps(command, ensure_ascii=False, indent=2))
    else:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
