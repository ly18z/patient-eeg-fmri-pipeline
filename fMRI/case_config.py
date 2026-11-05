"""Read the case settings shared by stage scripts (launched through run.py)."""
import json
import os
from pathlib import Path


class Case:
    def __init__(self, filename):
        self.filename = Path(filename).resolve()
        self.data = json.loads(self.filename.read_text(encoding='utf-8-sig'))
        self.root = Path(os.environ.get('FMRI_ROOT', self.data['root']))
        if not self.root.is_absolute():
            self.root = (self.filename.parent / self.root).resolve()
        tools = self.data.get('tools_directory', str(Path(__file__).resolve().parents[1] / 'tools'))
        self.tools = Path(os.environ.get('FMRI_TOOLS', tools))
        if not self.tools.is_absolute():
            self.tools = (self.filename.parent / self.tools).resolve()
        self.dicom = Path(os.environ.get('FMRI_DICOM', self.data['dicom']))
        if not self.dicom.is_absolute():
            self.dicom = (self.filename.parent / self.dicom).resolve()
        self.subject = self.data['subject']
        self.task_settings = self.data.get('task', {})
        self.task = self.task_settings.get('name', 'motor')
        self.condition = self.task_settings.get('condition', 'grasp')
        self.runs = [x['run'] for x in self.data['inputs'].get('task_runs', [])]
        self.models = self.tools / 'lesion_models'
        font = os.environ.get('FMRI_FONT') or self.data.get('font_file')
        self.font = (self.filename.parent / font).resolve() if font else None

    def configure_plotting(self):
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        if self.font:
            font_manager.fontManager.addfont(str(self.font))
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(self.font)).get_name()
        else:
            names = {f.name for f in font_manager.fontManager.ttflist}
            plt.rcParams['font.family'] = next((name for name in
                ['Noto Sans CJK SC', 'Microsoft YaHei', 'SimHei'] if name in names), 'DejaVu Sans')
        plt.rcParams['axes.unicode_minus'] = False

    def path(self, value):
        return self.root / value

    def input(self, name):
        return self.path(self.data['inputs'][name]['file'])

    def sidecar(self, name, extension):
        return Path(str(self.input(name)).removesuffix('.gz').removesuffix('.nii') + extension)

    def run(self, run):
        return next(x for x in self.data['inputs']['task_runs'] if x['run'] == run)

    def bold(self, run):
        return self.root / 'derivatives/fmriprep' / self.subject / 'func' / (
            f'{self.subject}_task-{self.task}_run-{run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')

    def events(self, run):
        import pandas as pd
        settings = self.task_settings | self.run(run)
        if settings.get('events'):
            return pd.read_csv(self.path(settings['events']), sep='\t')
        onsets = settings['onsets_seconds']
        return pd.DataFrame(dict(onset=onsets, duration=settings['duration_seconds'], trial_type=self.condition))

    def acquisition(self, path):
        import nibabel as nib
        image = nib.load(path)
        units = image.header.get_xyzt_units()[1]
        tr = float(image.header.get_zooms()[3]) * {'msec': .001, 'usec': .000001}.get(units, 1)
        sidecar = Path(str(path).removesuffix('.gz').removesuffix('.nii') + '.json')
        if sidecar.exists():
            tr = float(json.loads(sidecar.read_text(encoding='utf-8')).get('RepetitionTime', tr))
        return image.shape[3], tr


CASE = Case(os.environ['FMRI_CONFIG'])
