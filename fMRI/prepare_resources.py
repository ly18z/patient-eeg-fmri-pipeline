"""Prepare public optional MRI resources; downloads require --download."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile
import urllib.request
import zipfile

FSL = 'https://www.freesurfer.net/pub/dist/freesurfer/tutorial_packages_centos6/OSX/fsl_501/data/atlases/'
TF = 'https://templateflow.s3.amazonaws.com/'
SYNTH_COMMIT = 'f627b57b8f9291bfa870cb54cb21ddbeb4941d39'
TRUENET_COMMIT = 'd9e3118029e8be93c9930574751efd9e78533cf0'
MODELS = {'baseline': 'b693a650026359705688fbce409219c4dbb5d6be',
          'synth': 'ced72e683ed68d3d110b675714e1a07bac584948'}


def fetch(url, file):
    if file.exists():
        return file
    file.parent.mkdir(parents=True, exist_ok=True)
    print('Download:', url, '\n  ->', file, flush=True)
    partial = file.with_name(file.name+'.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'EEG-MRI-Pipeline/1.0'})
    with urllib.request.urlopen(request, timeout=60) as src, partial.open('wb') as dst:
        shutil.copyfileobj(src, dst)
    partial.replace(file)
    return file


def unpack_selected(archive, destination, wanted):
    with tarfile.open(archive) as tar:
        for member in tar:
            name = Path(member.name).name
            if member.isfile() and name in wanted:
                destination.mkdir(parents=True, exist_ok=True)
                target = destination/name
                if not target.exists():
                    with tar.extractfile(member) as src, target.open('wb') as dst:
                        shutil.copyfileobj(src, dst)


def atlases(case, cache, aal3):
    lesion = case.root/'work/lesion_atlas'
    cortical = case.root/'analysis_rest_dti/work/atlas'
    for name, folder, url in [
        ('HarvardOxford-sub-maxprob-thr25-1mm.nii.gz', lesion, FSL+'HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz'),
        ('HarvardOxford-Subcortical.xml', lesion, FSL+'HarvardOxford-Subcortical.xml'),
        ('HarvardOxford-cort-maxprob-thr25-1mm.nii.gz', cortical, FSL+'HarvardOxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz'),
        ('HarvardOxford-Cortical.xml', cortical, FSL+'HarvardOxford-Cortical.xml'),
        ('JHU-ICBM-labels-1mm.nii.gz', lesion, FSL+'JHU/JHU-ICBM-labels-1mm.nii.gz'),
        ('JHU-labels.xml', lesion, FSL+'JHU-labels.xml')]:
        fetch(url, folder/name)
    for name in ['tpl-MNI152NLin6Asym_res-02_T1w.nii.gz', 'tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz']:
        fetch(TF+'tpl-MNI152NLin6Asym/'+name, lesion/name)
    if aal3:
        dest = case.root/'analysis_glm_regions/atlas/AAL3_source'
        archive = fetch('https://gin.cnrs.fr/wp-content/uploads/AAL3v2_for_SPM12.tar.gz', cache/'AAL3v2.tar.gz')
        unpack_selected(archive, dest, {'AAL3v1_1mm.nii', 'AAL3v1_1mm.xml'})
        for name in ['tpl-MNIColin27_T1w.nii.gz', 'tpl-MNIColin27_desc-brain_mask.nii.gz']:
            fetch(TF+'tpl-MNIColin27/'+name, dest/name)


def dti(case, cache):
    dest = case.tools/'python/dti_linux_packages'
    dest.mkdir(parents=True, exist_ok=True)
    for package, version in [('dipy', '1.11.0'), ('trx-python', '0.6'), ('deepdiff', '9.1.0'), ('orderly-set', '5.5.0')]:
        with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json', timeout=60) as response:
            files = json.load(response)['urls']
        wheel = next(x for x in files if x['filename'].endswith('.whl') and
                     ('none-any' in x['filename'] or ('cp312-cp312' in x['filename'] and 'manylinux' in x['filename'] and 'x86_64' in x['filename'])))
        archive = fetch(wheel['url'], cache/wheel['filename'])
        with zipfile.ZipFile(archive) as wheel_file:
            wheel_file.extractall(dest)
    archive = fetch('https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-eddy-2401.2-h500b71b_4.tar.bz2', cache/'fsl-eddy.tar.bz2')
    unpack_selected(archive, case.tools/'fsl_eddy/bin', {'eddy_cpu'})
    (case.tools/'fsl_eddy/bin/eddy_cpu').chmod(0o755)


def models(case, cache):
    source = case.models/'synthstroke'
    for name in ['synthstroke_model.py', 'LICENSE']:
        fetch(f'https://raw.githubusercontent.com/liamchalcroft/SynthStroke/{SYNTH_COMMIT}/{name}', source/name)
    for name, revision in MODELS.items():
        dest = case.models/('synthstroke-'+name)
        for file in ['config.json', 'model.safetensors', 'README.md']:
            fetch(f'https://huggingface.co/liamchalcroft/synthstroke-{name}/resolve/{revision}/{file}', dest/file)
        (dest/'revision.txt').write_text(revision+'\n', encoding='utf-8')
    archive = fetch(f'https://codeload.github.com/v-sundaresan/truenet/zip/{TRUENET_COMMIT}', cache/'truenet-source.zip')
    dest = case.models/'truenet/source'
    with zipfile.ZipFile(archive) as z:
        prefix = z.namelist()[0]
        for member in z.namelist():
            relative = member.removeprefix(prefix)
            if not member.endswith('/') and (relative.startswith('truenet/') or Path(relative).name.startswith('LICENSE')):
                target = dest/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(member))
    archive = fetch('https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/noarch/fsl-truenet_data-2406.0-1.tar.bz2', cache/'truenet-models.tar.bz2')
    unpack_selected(archive, case.models/'truenet/model_data',
                    {f'Truenet_MWSC_FLAIR_T1_{plane}.pth' for plane in ['axial', 'sagittal', 'coronal']})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--groups', nargs='+', choices=['atlases', 'dti', 'models'])
    p.add_argument('--download', action='store_true', help='Explicitly start downloading public resources')
    args = p.parse_args()
    os.environ['FMRI_CONFIG'] = str(args.config.resolve())
    from case_config import CASE
    modules = set(CASE.data.get('modules', []))
    groups = args.groups if args.groups is not None else (
        (['atlases'] if modules & {'regions', 'rest', 'dti', 'lesion'} else [])+
        (['dti'] if 'dti' in modules else [])+(['models'] if 'lesion' in modules else []))
    print('Selected resources:', ', '.join(groups) or 'none')
    print('Tools:', CASE.tools, '\nCase:', CASE.root)
    if not args.download:
        print('No downloads started. Add --download after checking the selected groups.')
        return
    cache = CASE.tools/'downloads'
    if 'atlases' in groups:
        atlases(CASE, cache, 'regions' in modules)
    if 'dti' in groups:
        dti(CASE, cache)
    if 'models' in groups:
        models(CASE, cache)
    print('Resources ready.')


if __name__ == '__main__':
    main()
