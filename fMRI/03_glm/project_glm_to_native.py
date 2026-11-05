"""Map existing MNI GLM statistics to the patient's original T1 grid.

Continuous maps: linear interpolation. Already-corrected maps: nearest neighbor,
preserving the MNI-defined significance selection without retesting native voxels.
Offline HTML viewer packs the patient's actual T1 and mapped corrected maps.
"""
from case_config import CASE
from pathlib import Path
import os, json, subprocess, time, gzip, base64, shutil
os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilearn import plotting
ROOT = CASE.root
OUT=ROOT/'derivatives/glm_native'
QC=ROOT/'qc/glm_native'
LOG=ROOT/'logs'
for p in [OUT,QC,LOG]:p.mkdir(exist_ok=True,parents=True)
ANAT=ROOT/f'derivatives/fmriprep/{CASE.subject}/anat'
REF=ROOT/f'bids/{CASE.subject}/anat/{CASE.subject}_T1w.nii.gz'
PRE=ANAT/f'{CASE.subject}_desc-preproc_T1w.nii.gz'
INV=ANAT/f'{CASE.subject}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5'
T1MNI=ANAT/f'{CASE.subject}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz'
MASKMNI=ANAT/f'{CASE.subject}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz'
SUMMARY=json.loads((ROOT/'derivatives/glm/glm_summary.json').read_text(encoding='utf-8'))
START=time.time()
commands=[]
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def status(state,**extra):
    write(LOG/'glm_native_status.json',dict(status=state,elapsed_seconds=round(time.time()-START,1),**extra))
    print(state,json.dumps(extra),flush=True)

def warp(src, dst, interpolation):
    cmd=['antsApplyTransforms','-d','3','-e','0','--float','1','-i',str(src),'-r',str(REF),
        '-o',str(dst),'-n',interpolation,'-t',str(INV),'--default-value','0']
    subprocess.run(cmd,check=True,capture_output=True,text=True)
    commands.append(cmd)
    im=nib.load(dst)
    ref=nib.load(REF)
    assert im.shape==ref.shape and np.allclose(im.affine,ref.affine,atol=1e-4)
    assert np.isfinite(im.get_fdata(dtype=np.float32)).all()
    write(dst.with_suffix('').with_suffix('.json'),dict(Sources=[src.relative_to(ROOT).as_posix()],
        ReferenceImage=REF.relative_to(ROOT).as_posix(),Transform=INV.relative_to(ROOT).as_posix(),
        Interpolation=interpolation,Description='Inverse-warped existing GLM result for native anatomy display; GLM and multiple-comparison correction not refitted'))
    return im

def main():
    status('PROJECTING')
    ref=nib.load(REF)
    pre=nib.load(PRE)
    assert ref.shape==pre.shape and np.allclose(ref.affine,pre.affine)
    assert nib.aff2axcodes(ref.affine)==('R','A','S')
    shutil.copy2(REF,OUT/f'{CASE.subject}_space-T1w_desc-original_T1w.nii.gz')
    mask_native=nib.load(ANAT/f'{CASE.subject}_desc-brain_mask.nii.gz').get_fdata()>.5
    recovered=warp(T1MNI,QC/f'{CASE.subject}_desc-inverseWarpCheck_T1w.nii.gz','Linear').get_fdata(dtype=np.float32)
    recovered_mask=warp(MASKMNI,QC/f'{CASE.subject}_desc-inverseWarpCheck_mask.nii.gz','NearestNeighbor').get_fdata()>.5
    dice=float(2*np.sum(mask_native&recovered_mask)/(mask_native.sum()+recovered_mask.sum()))
    interior=mask_native&recovered_mask
    correlation=float(np.corrcoef(pre.get_fdata(dtype=np.float32)[interior],recovered[interior])[0,1])
    assert dice>.85 and correlation>.75,('Check transform direction/alignment',dice,correlation)
    common=next((ROOT/'derivatives/glm').glob('*desc-common_mask.nii.gz'))
    warp(common,OUT/f'{CASE.subject}_task-{CASE.task}_space-T1w_desc-common_mask.nii.gz','NearestNeighbor')
    results=[]
    viewer_maps={}
    for label, short, st in [(f"{CASE.subject}_task-{CASE.task}_run-{r['model']['run']}", f"Run {r['model']['run']}", r['statistics']) for r in SUMMARY['runs']] + [(f'{CASE.subject}_task-{CASE.task}_combined', 'Combined', SUMMARY['combined'])]:
        status('PROJECTING_MAPS',result=short)
        source=ROOT/'derivatives/glm'/label
        dest=OUT/label
        dest.mkdir(exist_ok=True)
        maps={}
        for suffix,interp in [('contrast-taskVsRest_desc-zmap','Linear'),('contrast-taskVsRest_desc-effect','Linear'),('desc-fdr05_zmap','NearestNeighbor'),('desc-fwe05_zmap','NearestNeighbor')]:
            target=dest/f'{label}_space-T1w_{suffix}.nii.gz'
            maps[suffix]=warp(source/f'{label}_{suffix}.nii.gz',target,interp)
        selected=maps['desc-fdr05_zmap']
        selected_data=selected.get_fdata(dtype=np.float32)
        for mode,cuts,size in [('z',[-10,0,10,20,30,40,50,60,70],(21,4)),('ortho',(0,-20,50),(12,5))]:
            fig=plt.figure(figsize=size)
            display=plotting.plot_stat_map(selected,bg_img=str(REF),display_mode=mode,cut_coords=cuts,
                threshold=max((st['fdr05']['z_threshold'] or 0.)-1e-6,0.),resampling_interpolation='nearest',cmap='cold_hot',symmetric_cbar=True,vmax=10,black_bg=False,figure=fig,
                title=f'{short} | patient native T1 | MNI-defined FDR q<0.05')
            display.savefig(str(QC/f'{label}_native_fdr05_{"axial" if mode=="z" else "ortho"}.png'),dpi=160);display.close();plt.close(fig)
        for correction,tag in [('fdr05','FDR'),('fwe05','FWE')]:
            v=maps[f'desc-{correction}_zmap'].get_fdata(dtype=np.float32)
            quantized=np.rint(np.clip(v,-10,10)*10).astype(np.int8)
            viewer_maps[f'{short}_{tag}']=base64.b64encode(gzip.compress(quantized.tobytes(order='C'),mtime=0)).decode()
        results.append(dict(label=label,reference_grid_shape=list(ref.shape),
            mapped_positive_FDR_native_grid_voxels=int(np.sum(selected_data>0)),
            mapped_negative_FDR_native_grid_voxels=int(np.sum(selected_data<0)),
            note='Native-grid counts are resampling descriptions, not new statistical tests; original inference remains in MNI space'))
    raw=ref.get_fdata(dtype=np.float32)
    lo,hi=np.percentile(raw[mask_native],[1,99.5])
    gray=np.rint(np.clip((raw-lo)/(hi-lo),0,1)*255).astype(np.uint8)
    gray_b64=base64.b64encode(gzip.compress(gray.tobytes(order='C'),mtime=0)).decode()
    check_img=nib.Nifti1Image(recovered_mask.astype(np.uint8),ref.affine)
    disp=plotting.plot_roi(check_img,bg_img=str(PRE),display_mode='z',cut_coords=[-10,0,10,20,30,40,50,60,70],
        title=f'Inverse-transform check | brain-mask Dice={dice:.3f}, T1 r={correlation:.3f}',alpha=.25)
    disp.savefig(str(QC/'native_inverse_transform_check.png'),dpi=160)
    disp.close()
    payload=dict(shape=list(ref.shape),affine=ref.affine.tolist(),t1=gray_b64,maps=viewer_maps)
    template=(Path(__file__).with_name('native_glm_viewer_template.html')).read_text(encoding='utf-8')
    (ROOT/'GLM_患者原生影像.html').write_text(template.replace('__DATA_JSON__',json.dumps(payload,separators=(',',':'))),encoding='utf-8')
    config=dict(status='PROJECTED_VISUAL_REVIEW_PENDING',source_space='MNI152NLin2009cAsym',target_space='patient original T1w grid',
        reference=REF.relative_to(ROOT).as_posix(),reference_shape=list(ref.shape),reference_spacing_mm=list(map(float,ref.header.get_zooms())),
        transform=INV.relative_to(ROOT).as_posix(),transform_direction='MNI to native T1w, no extra inversion',
        continuous_interpolation='Linear',corrected_map_interpolation='NearestNeighbor',
        thresholding='No native-space retesting or rethresholding; preserve existing MNI corrected maps',
        validation=dict(recovered_T1_intensity_correlation=correlation,brain_mask_Dice=dice,reference_grid_matched=True,finite_pixels=True,nearest_neighbor_preserves_existing_Z_values=True),
        results=results,commands=commands,elapsed_seconds=round(time.time()-START,1))
    write(OUT/'projection_summary.json',config)
    status('PROJECTED_VISUAL_REVIEW_PENDING',mask_Dice=dice,T1_correlation=correlation)
if __name__ == '__main__':
    main()
