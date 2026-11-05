"""Run the MRI pipeline from its own configuration."""
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

    case = config.copy()
    case['root'], case['dicom'] = str(path(case['root'])), str(path(case['dicom']))
    root = Path(case['root'])
    modules = set(config['modules'])
    unknown = modules - {'convert', 'bids', 'preprocess', 'glm', 'native', 'curves', 'regions', 'rest', 'dti', 'lesion'}
    if unknown:
        raise ValueError('Unknown MRI modules: '+', '.join(sorted(unknown)))
    def mri_step(name):
        command('fmri.'+name, [python, '-X', 'utf8', '-B', HERE/'run.py', name,
                              '--config', filename])
    if 'convert' in modules:
        mri_step('convert')
    if 'bids' in modules:
        mri_step('prepare_bids')
    if 'preprocess' in modules:
        for name in ['preprocess', 'qc_preprocessing', 'finalize_preprocessing']:
            mri_step(name)
        if config.get('manual_qc', True):
            pause('fmri.review_preprocessing', '检查脑提取、组织分割、头动、覆盖及BOLD-T1/模板配准。',
                  [root/'derivatives/fmriprep'/f'{case["subject"]}.html', root/'预处理结果说明.md'])
    if modules & {'glm', 'native', 'curves', 'regions'}:
        mri_step('run_task_glm')
        mri_step('finalize_task_glm')
    if 'native' in modules:
        mri_step('project_glm_to_native')
    if 'curves' in modules:
        mri_step('plot_glm_bold_curves')
        mri_step('attach_glm_bold_curves')
    if modules & {'regions', 'rest', 'dti', 'lesion'}:
        mri_step('register_lesion_atlas')
        if modules & {'dti', 'lesion'}:
            mri_step('add_jhu_lesion_reference')
        if config.get('manual_qc', True):
            pause('fmri.review_registration', '检查模板/图谱到患者T1的局部配准，尤其是目标通路及病灶附近。',
                  [root/'qc/lesion'])
    if 'regions' in modules:
        for name in ['annotate_glm_regions', 'register_aal3', 'append_aal3_annotations', 'report_glm_regions']:
            mri_step(name)
    if modules & {'rest', 'dti'}:
        mri_step('prepare_motor_rois')
    if 'rest' in modules:
        mri_step('run_rest_connectivity')
    if 'dti' in modules:
        for name in ['audit_dwi_gradients', 'run_dwi_preprocessing', 'fit_dti_and_tracks', 'finalize_rest_dti']:
            mri_step(name)
        if config.get('manual_qc', True):
            pause('fmri.review_dti', '检查DWI-T1配准、扩散指标及候选CST的走行与目标约束。',
                  [root/'analysis_rest_dti/qc'])
    if {'rest', 'dti'} <= modules:
        mri_step('report_rest_dti')
    if 'lesion' in modules:
        for name in ['prepare_lesion_images', 'run_chronic_stroke_models', 'run_truenet_wmh', 'report_lesion_candidates']:
            mri_step(name)
        if config.get('manual_qc', True):
            pause('fmri.review_lesion', '结合T1/FLAIR复核模型候选；此节点不会自动把候选确认为病灶。',
                  [root/'候选病变影像查看.html', root/'病灶候选定位报告.html'])
    return run, steps


if __name__ == "__main__":
    entry(make_plan, __doc__)
