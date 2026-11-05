"""Map a standard JHU white matter reference using the fitted MNI6 warp."""
from case_config import CASE
import json, subprocess, xml.etree.ElementTree as ET
import numpy as np,nibabel as nib
ROOT = CASE.root
A=ROOT/'work/lesion_atlas'
OUT=ROOT/'derivatives/lesion/anatomy'
src=A/'JHU-ICBM-labels-1mm.nii.gz'
ref=ROOT/f'bids/{CASE.subject}/anat/{CASE.subject}_T1w.nii.gz'
dest=OUT/f'{CASE.subject}_space-T1w_atlas-JHUWhiteMatter_dseg.nii.gz'
cmd=['antsApplyTransforms','-d','3','--float','1','-i',str(src),'-r',str(ref),'-o',str(dest),'-n','NearestNeighbor',
 '-t',str(OUT/'from-MNI6_to-T1w_1Warp.nii.gz'),'-t',str(OUT/'from-MNI6_to-T1w_0GenericAffine.mat')]
subprocess.run(cmd,check=True)
im=nib.load(dest)
assert im.shape==nib.load(ref).shape and np.allclose(im.affine,nib.load(ref).affine)
els=ET.parse(A/'JHU-labels.xml').findall('.//label')
first=els[0].text.lower()
offset=0 if any(x in first for x in ['unclassified','background','unknown']) else 1
labels={int(e.attrib['index'])+offset:e.text.strip() for e in els}
(OUT/'jhu_labels.json').write_text(json.dumps(labels,indent=2),encoding='utf-8')
(OUT/'jhu_registration.json').write_text(json.dumps(dict(status='REGISTERED',source='FSL JHU ICBM labels, standard-space reference',
 command=cmd,label_offset=offset,interpretation='Warped population white matter region reference; not patient diffusion tractography'),indent=2),encoding='utf-8')
print('JHU_REFERENCE_READY',labels[19],labels[20])
