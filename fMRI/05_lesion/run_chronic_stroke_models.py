"""Offline native-T1 chronic stroke candidate inference with pinned SynthStroke models.

Uses author test.py preprocessing with correct affine metadata before spatial
transforms. The upstream convenience preprocess_image attaches affine too late,
so spatial preprocessing is explicit here. No patient image is sent to a server.
"""
from case_config import CASE
import os, sys, json, time, copy, importlib.metadata as md
ROOT = CASE.root
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['WANDB_DISABLED']='true'
sys.path.insert(0,str(CASE.models/'synthstroke'))
import numpy as np,nibabel as nib,torch,monai as mn
from monai.data import MetaTensor
from scipy import ndimage
from safetensors.torch import load_file
from synthstroke_model import SynthStrokeModel,ModelConfig
OUT=ROOT/'derivatives/lesion'
OUT.mkdir(exist_ok=True,parents=True)
LOG=ROOT/'logs'
START=time.time()
torch.set_num_threads(6)
torch.manual_seed(20261026)
T1=ROOT/f'bids/{CASE.subject}/anat/{CASE.subject}_T1w.nii.gz'
BRAIN=ROOT/f'derivatives/fmriprep/{CASE.subject}/anat/{CASE.subject}_desc-brain_mask.nii.gz'
ref=nib.load(T1)
brain=nib.load(BRAIN).get_fdata()>.5
assert brain.shape==ref.shape and np.allclose(nib.load(BRAIN).affine,ref.affine)
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def status(phase,**kw):
 write(LOG/'lesion_models_status.json',dict(status=phase,elapsed_seconds=round(time.time()-START,1),**kw));print(phase,json.dumps(kw),flush=True)
def save(data,p,dtype=np.float32):
 hdr=ref.header.copy();hdr.set_data_dtype(dtype);nib.save(nib.Nifti1Image(np.asarray(data,dtype=dtype),ref.affine,hdr),p)
prep=mn.transforms.Compose([
 mn.transforms.LoadImaged(keys='img',image_only=True),mn.transforms.EnsureChannelFirstd(keys='img'),
 mn.transforms.Orientationd(keys='img',axcodes='RAS'),mn.transforms.Spacingd(keys='img',pixdim=(1,1,1),mode='bilinear'),
 mn.transforms.HistogramNormalized(keys='img'),mn.transforms.NormalizeIntensityd(keys='img',nonzero=False,channel_wise=True)])
image=prep({'img':str(T1)})['img'].to(dtype=torch.float32)
assert isinstance(image,MetaTensor) and len(image.applied_operations)>=2
all_results=[]
masks={}
for variant in ['baseline', 'synth']:
    dest=OUT/f'model-{variant}'
    dest.mkdir(exist_ok=True)
    assets=CASE.models/('synthstroke-'+variant)
    status('LOADING_MODEL',variant=variant)
    cfg=ModelConfig.from_dict(json.loads((assets/'config.json').read_text(encoding='utf-8')))
    model=SynthStrokeModel(cfg)
    state=load_file(str(assets/'model.safetensors'))
    if not any(k.startswith('unet.') for k in state):state={'unet.'+k:v for k,v in state.items()}
    model.load_state_dict(state,strict=True)
    model.eval()
    del state
    status('INFERENCE',variant=variant,input_shape=list(image.shape),patch=128,overlap=.5,tta=False)
    with torch.inference_mode():
     logits=mn.inferers.sliding_window_inference(image[None],roi_size=(128,128,128),sw_batch_size=1,
       predictor=model,overlap=.5,mode='gaussian',progress=True,sw_device='cpu',device='cpu')
     output=MetaTensor(logits[0].cpu(),affine=image.affine,meta=copy.deepcopy(image.meta),applied_operations=copy.deepcopy(image.applied_operations))
     with mn.transforms.utils.allow_missing_keys_mode(prep):native=prep.inverse({'img':output})['img']
     probs=torch.softmax(native,dim=0).cpu().numpy()
    assert probs.shape[1:]==ref.shape and np.allclose(native.affine.numpy(),ref.affine,atol=1e-4),(probs.shape,native.affine)
    assert np.isfinite(probs).all() and np.allclose(probs.sum(axis=0),1,atol=1e-5)
    labels=probs.argmax(axis=0).astype(np.uint8)
    lesion_class=1 if cfg.out_channels==2 else 5
    prob=probs[lesion_class]
    raw=labels==lesion_class
    mask=raw&brain
    masks[variant]=mask
    save(prob,dest/f'{CASE.subject}_space-T1w_model-{variant}_desc-lesion_probseg.nii.gz')
    save(labels,dest/f'{CASE.subject}_space-T1w_model-{variant}_desc-multiclass_dseg.nii.gz',np.uint8)
    save(raw,dest/f'{CASE.subject}_space-T1w_model-{variant}_desc-rawcandidate_mask.nii.gz',np.uint8)
    save(mask,dest/f'{CASE.subject}_space-T1w_model-{variant}_desc-braincandidate_mask.nii.gz',np.uint8)
    labs,n=ndimage.label(mask,structure=ndimage.generate_binary_structure(3,1))
    components=[]
    volume=abs(np.linalg.det(ref.affine[:3,:3]))
    for j in range(1,n+1):
     ijk=np.argwhere(labs==j);peak=ijk[np.argmax(prob[tuple(ijk.T)])];centroid=nib.affines.apply_affine(ref.affine,ijk.mean(axis=0))
     components.append(dict(component_id=j,voxels=len(ijk),volume_ml=float(len(ijk)*volume/1000),
      centroid_native_mm=centroid.tolist(),peak_probability=float(prob[tuple(peak)]),
      mean_probability=float(prob[labs==j].mean()),laterality='left' if centroid[0]<0 else 'right'))
    components.sort(key=lambda x:x['voxels'],reverse=True)
    result=dict(variant=variant,model_repo='liamchalcroft/synthstroke-'+variant,
     model_revision=(assets/'revision.txt').read_text(encoding='utf-8').strip() if (assets/'revision.txt').exists() else 'user-supplied',
     lesion_class=lesion_class,selection='native-space argmax class; then intersect with existing T1 brain mask',
     raw_candidate_voxels=int(raw.sum()),brain_candidate_voxels=int(mask.sum()),
     brain_candidate_volume_ml=float(mask.sum()*volume/1000),removed_extracranial_voxels=int((raw&~brain).sum()),
     components=components,inputs={'T1':str(T1.relative_to(ROOT)),'brain_mask':str(BRAIN.relative_to(ROOT))},
     preprocessing='author test.py: RAS, 1mm spacing, histogram equalization, whole-image intensity z-normalization; native affine present before transforms',
     inference={'patch':128,'overlap':.5,'blending':'gaussian logits','TTA':False,'device':'CPU','threads':6},
     native_grid_verified=True)
    write(dest/'model_result.json',result)
    all_results.append(result)
    status('MODEL_COMPLETE',variant=variant,volume_ml=result['brain_candidate_volume_ml'],components=len(components))
    del model,logits,native,probs,labels,prob,raw,labs
consensus=masks['baseline']&masks['synth']
union=masks['baseline']|masks['synth']
save(consensus,OUT/f'{CASE.subject}_space-T1w_desc-modelAgreement_mask.nii.gz',np.uint8)
save(union,OUT/f'{CASE.subject}_space-T1w_desc-modelUnion_mask.nii.gz',np.uint8)
denom=masks['baseline'].sum()+masks['synth'].sum()
summary=dict(status='INFERENCE_COMPLETE_REVIEW_PENDING',clinical_context=CASE.data.get('clinical_context',{}),
 models=all_results,model_agreement_dice=float(2*consensus.sum()/denom) if denom else None,
 agreement_voxels=int(consensus.sum()),union_voxels=int(union.sum()),
 note='Agreement/union are model-comparison masks, not validated final lesion ground truth. No candidate was manually edited.',
 software={p:md.version(p) for p in ['torch','monai','numpy','scipy','nibabel','safetensors','huggingface_hub']},elapsed_seconds=time.time()-START)
write(OUT/'lesion_model_summary.json',summary)
status('INFERENCE_COMPLETE_REVIEW_PENDING',summary='derivatives/lesion/lesion_model_summary.json')
