"""Convert a DICOM series to NIfTI/JSON, retaining diffusion gradients."""
import argparse
import json
from pathlib import Path
import subprocess
import nibabel as nib
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dicom',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--dcm2niix',default='dcm2niix',help='Executable on PATH or its full path')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();source=a.dicom.resolve()
    if a.output.resolve().is_relative_to(source):p.error('Output must stay outside the read-only DICOM tree')
    command=[str(a.dcm2niix),'-b','y','-ba','y','-z','y','-f','series-%s','-o',str(a.output.resolve()),str(source)]
    if a.dry_run:print(json.dumps(command,ensure_ascii=False,indent=2));return
    a.output.mkdir(parents=True)
    with (a.output/'dcm2niix.log').open('w',encoding='utf-8') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    outputs=list(a.output.glob('*.nii.gz'))+list(a.output.glob('*.nii'))
    if not outputs:raise RuntimeError('Converter returned no NIfTI')
    summary=[]
    for file in outputs:
        image=nib.load(file);data=image.get_fdata(dtype=np.float32)
        if data.ndim not in [3,4] or not np.isfinite(data).all() or abs(np.linalg.det(image.affine[:3,:3]))<1e-12:
            raise ValueError('Invalid scalar 3D/4D NIfTI; inspect source and converter output')
        stem=file.name.removesuffix('.gz').removesuffix('.nii');bv=a.output/(stem+'.bvec');ba=a.output/(stem+'.bval')
        if bv.exists() or ba.exists():
            values=np.atleast_1d(np.loadtxt(ba));vectors=np.loadtxt(bv)
            if len(image.shape)!=4 or len(values)!=image.shape[3] or vectors.shape!=(3,len(values)):
                raise ValueError('DWI gradient/image volume count mismatch')
            if not np.isfinite(values).all() or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors[:,values>50],axis=0),1,atol=1e-3):
                raise ValueError('Invalid diffusion gradient values')
        summary.append(dict(file=file.name,shape=image.shape,zooms=list(map(float,image.header.get_zooms())),
            orientation=nib.aff2axcodes(image.affine),affine=image.affine.tolist(),gradients_present=bv.exists()))
    record=dict(status='converted',command=command,source=str(source),outputs=summary)
    (a.output/'conversion_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(a.output)


if __name__=='__main__':main()
