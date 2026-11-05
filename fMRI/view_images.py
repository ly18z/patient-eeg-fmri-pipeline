"""Local offline orthogonal slice viewer for a scalar NIfTI and aligned overlays."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--background',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--space',required=True,help='Explicit space label; e.g. patient native T1w')
    p.add_argument('--overlay',type=Path,help='Aligned signed GLM/statistic/scalar map')
    p.add_argument('--lesion',type=Path,help='Aligned candidate mask/probability map')
    p.add_argument('--lesion-label',default='Automatic lesion candidate (unconfirmed)')
    p.add_argument('--volume',type=int,default=0);p.add_argument('--stride',type=int,default=2)
    p.add_argument('--threshold',type=float,default=0,help='Display |overlay| threshold, not a statistical correction')
    a=p.parse_args()
    original=nib.load(a.background)
    if len(original.shape)==4:
        reference=nib.Nifti1Image(np.asarray(original.dataobj[...,a.volume]),original.affine)
    elif len(original.shape)==3:reference=original
    else:raise ValueError('Expected scalar 3D/4D background')
    canonical=nib.as_closest_canonical(reference);data=canonical.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():raise ValueError('Nonfinite background')
    def load_overlay(path):
        if path is None:return None
        image=nib.load(path)
        if image.shape!=reference.shape or not np.allclose(image.affine,reference.affine,atol=1e-5):
            raise ValueError('Overlay grid/affine differs. Register/resample using a verified transform first.')
        value=nib.as_closest_canonical(image).get_fdata(dtype=np.float32)
        if not np.isfinite(value).all():raise ValueError('Nonfinite overlay')
        return value
    overlay=load_overlay(a.overlay);lesion=load_overlay(a.lesion)
    low,high=np.percentile(data,[1,99.5]);high=max(high,low+1e-9)
    scale=float(max(np.max(abs(overlay)),1e-12)) if overlay is not None else 1
    frames={};zooms=canonical.header.get_zooms()[:3]
    for axis,name in enumerate(['sagittal','coronal','axial']):
        frames[name]=[];other=[i for i in range(3) if i!=axis]
        indices=list(range(0,data.shape[axis],a.stride))
        if data.shape[axis]-1 not in indices:indices.append(data.shape[axis]-1)
        for index in indices:
            slice2=lambda x:np.rot90(np.take(x,index,axis=axis))
            gray=np.clip((slice2(data)-low)/(high-low),0,1)
            rgb=np.repeat(gray[:,:,None],3,axis=2)
            if overlay is not None:
                z=slice2(overlay);active=(abs(z)>a.threshold)&(z!=0)
                color=np.zeros_like(rgb);color[z>0]=[1,.15,.05];color[z<0]=[.05,.4,1]
                alpha=.75*np.sqrt(np.clip(abs(z)/scale,0,1))*active
                rgb=rgb*(1-alpha[:,:,None])+color*alpha[:,:,None]
            if lesion is not None:
                mask=slice2(lesion)>.5;edge=mask&~binary_erosion(mask)
                rgb[edge]=[1,1,0]
            image=Image.fromarray(np.uint8(np.clip(rgb,0,1)*255))
            width=data.shape[other[0]]*zooms[other[0]];height=data.shape[other[1]]*zooms[other[1]]
            ratio=384/max(width,height);image=image.resize((max(1,round(width*ratio)),max(1,round(height*ratio))))
            memory=BytesIO();image.save(memory,format='PNG')
            voxel=(np.array(data.shape)-1)/2;voxel[axis]=index
            xyz=nib.affines.apply_affine(canonical.affine,voxel)
            frames[name].append(dict(index=index,center_xyz_mm=xyz.tolist(),image='data:image/png;base64,'+base64.b64encode(memory.getvalue()).decode()))
    metadata=dict(space=a.space,volume=a.volume,stride=a.stride,background=str(a.background),
        lesion_label=a.lesion_label if a.lesion else 'No lesion mask supplied',overlay_display_threshold=a.threshold,
        note='Coordinates are in the declared physical space. Native RAS is not MNI. Display threshold is not statistical significance.')
    payload=json.dumps(dict(metadata=metadata,frames=frames),ensure_ascii=True).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><title>Local MRI viewer</title><style>body{background:#14202c;color:#eef4fa;font:16px system-ui;padding:20px}main{display:flex;gap:20px;flex-wrap:wrap}section{background:#213243;padding:15px}img{display:block;max-width:100%;height:384px;object-fit:contain}input[type=range]{width:100%}pre{white-space:pre-wrap}</style><h1>本地MRI多平面查看</h1><p id="notes"></p><main></main><pre id="meta"></pre><script>const data='''+payload+''';document.querySelector('#notes').textContent='红：正向叠加；蓝：负向叠加；黄：'+data.metadata.lesion_label;document.querySelector('#meta').textContent=JSON.stringify(data.metadata,null,2);for(const [name,frames] of Object.entries(data.frames)){const s=document.createElement('section');s.innerHTML='<h2></h2><img><input type="range"><p></p>';s.querySelector('h2').textContent=name;const slider=s.querySelector('input');slider.min=0;slider.max=frames.length-1;slider.value=Math.floor(frames.length/2);function update(){const f=frames[Number(slider.value)];s.querySelector('img').src=f.image;s.querySelector('p').textContent='slice '+f.index+' | plane center xyz mm: '+f.center_xyz_mm.map(x=>x.toFixed(1)).join(', ')}slider.oninput=update;update();document.querySelector('main').appendChild(s)}</script>'''
    a.output.mkdir(parents=True);(a.output/'index.html').write_text(page,encoding='utf-8')
    metadata['inputs']=[str(x.resolve()) for x in [a.background,a.overlay,a.lesion] if x]
    (a.output/'viewer_manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(a.output/'index.html')


if __name__=='__main__':main()
