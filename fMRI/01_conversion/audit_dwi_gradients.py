from case_config import CASE
import pydicom,json
R = CASE.root
S=CASE.dicom
cur=None
directions={}
read=0
for x in pydicom.dcmread(S/'DICOMDIR').DirectoryRecordSequence:
    if x.DirectoryRecordType=='SERIES':cur=int(getattr(x,'SeriesNumber',-1))
    elif x.DirectoryRecordType=='IMAGE' and cur==CASE.data['inputs']['dwi']['series']:
        v=x.ReferencedFileID;p=S.joinpath(*(v.split('\\') if isinstance(v,str) else list(v)))
        d=pydicom.dcmread(p,stop_before_pixels=True,specific_tags=['DiffusionBValue','DiffusionGradientOrientation','ImageOrientationPatient'])
        g=list(map(float,d.DiffusionGradientOrientation));key=tuple(round(a,5) for a in g)
        directions.setdefault(key,{'b':float(d.DiffusionBValue),'LPS':g,'orientation_LPS':list(map(float,d.ImageOrientationPatient))});read+=1
out={'unique_gradients':list(directions.values()),'DICOM_headers_read':read,'scope':'Unique gradient directions, not source/NIfTI volume order'}
(R/'analysis_rest_dti/work').mkdir(parents=True,exist_ok=True)
(R/'analysis_rest_dti/work/gradient_dicom_unique.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('Unique DICOM gradients',len(directions),'headers',read)
