"""Organize configured images, sidecars and task events as BIDS."""
from case_config import CASE
from pathlib import Path
import json
import shutil

root = CASE.root
bids = root/'bids'
records = []
scans = [(name, item) for name, item in CASE.data['inputs'].items() if name != 'task_runs']
scans += [('task', item) for item in CASE.data['inputs'].get('task_runs', [])]
strip = {'AcquisitionTime', 'InstitutionName', 'InstitutionalDepartmentName', 'StationName',
         'DeviceSerialNumber', 'ProcedureStepDescription', 'StudyDescription', 'BidsGuess'}
for name, item in scans:
    source = str(root/item['file']).removesuffix('.gz').removesuffix('.nii')
    suffix = {'t1': 'T1w', 'flair': 'FLAIR', 'dwi': 'dwi', 'rest': 'task-rest_bold'}.get(name)
    if name == 'task':
        suffix = f'task-{CASE.task}_run-{item["run"]}_bold'
    folder = 'anat' if name in ['t1', 'flair'] else 'dwi' if name == 'dwi' else 'func'
    target = bids/CASE.subject/folder/f'{CASE.subject}_{suffix}'
    target.parent.mkdir(parents=True, exist_ok=True)
    for extension in ['.nii.gz', '.bval', '.bvec'] if name == 'dwi' else ['.nii.gz']:
        shutil.copy2(source+extension, str(target)+extension)
    meta = json.loads(Path(source+'.json').read_text(encoding='utf-8-sig'))
    meta = {k: v for k, v in meta.items() if k not in strip}
    if name in ['task', 'rest']:
        meta['TaskName'] = CASE.task if name == 'task' else 'rest'
    Path(str(target)+'.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    if name == 'task':
        CASE.events(item['run']).to_csv(str(target).removesuffix('_bold')+'_events.tsv', sep='\t', index=False)
    records.append(dict(source=source, bids_file=str(target.relative_to(bids))))
(bids/'dataset_description.json').write_text(json.dumps(dict(Name=root.name, BIDSVersion='1.11.1', DatasetType='raw'), indent=2), encoding='utf-8')
(bids/'participants.tsv').write_text(f'participant_id\n{CASE.subject}\n', encoding='utf-8')
(root/'logs').mkdir(exist_ok=True)
(root/'logs/bids_preparation.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
print(bids)
