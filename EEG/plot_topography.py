"""Plot Morlet band ERD scalp maps: mean of per-attempt dB, not mean TFR colors."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import mne
import numpy as np
from mne.channels.layout import _find_topomap_coords
from erd import load_results
from ica_common import write_json, write_csv


def head_outlines(sphere):
    """Use one sphere for the head, interpolation mask, nose and ears."""
    x,y,_,r=sphere;theta=np.linspace(0,2*np.pi,101)
    head=(x+r*np.cos(theta),y+r*np.sin(theta))
    ear_y=np.linspace(-.16,.16,30)*r+y
    ear_offset=np.sin(np.linspace(0,np.pi,30))*.08*r
    return dict(head=head,nose=(x+r*np.array([-.1,0,.1]),y+r*np.array([.995,1.12,.995])),
        ear_left=(x-r-ear_offset,ear_y),ear_right=(x+r+ear_offset,ear_y),
        mask_pos=head,clip_radius=(r,r),clip_origin=(x,y))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--morlet',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--montage',required=True)
    a=p.parse_args()
    z,record=load_results(a.morlet);channels=z['channels'].tolist()
    info=mne.create_info(channels,250,'eeg');info.set_montage(mne.channels.read_custom_montage(a.montage) if Path(a.montage).is_file() else a.montage)
    xyz=np.array([c['loc'][:3] for c in info['chs']])
    radius,origin,_=mne.bem.fit_sphere_to_headshape(info,dig_kinds='eeg',units='m',verbose=False)
    sphere=np.r_[origin,radius];outlines=head_outlines(sphere)
    positions=_find_topomap_coords(info,np.arange(len(channels)),sphere=sphere)
    scale=min(1.,.95*radius/np.linalg.norm(positions-origin[:2],axis=1).max())
    positions=origin[:2]+(positions-origin[:2])*scale
    n=len(z['attempt_clusters']);bands=z['bands'].tolist()
    effect=z['effect_db']
    mean=effect.mean(axis=0) if n else np.full((len(channels),len(bands)),np.nan)
    a.output.mkdir(parents=True)
    rows=[dict(channel=ch,band=band,n_attempts=n,mean_db=float(mean[j,k]) if n else None)
          for j,ch in enumerate(channels) for k,band in enumerate(bands)]
    write_csv(a.output/'topography_values.csv',rows,['channel','band','n_attempts','mean_db'])
    np.savez_compressed(a.output/'topography_values.npz',mean_db=mean,channels=channels,bands=bands,positions_m=xyz,
        display_positions_2d_m=positions,display_position_scale=scale,sphere_m=sphere,attempt_clusters=z['attempt_clusters'])
    fig,axes=plt.subplots(1,len(bands),figsize=(len(bands)*3.4,4.2),squeeze=False,layout='constrained')
    for k,ax in enumerate(axes.flat):
        if n:mne.viz.plot_topomap(mean[:,k],positions,axes=ax,show=False,cmap='RdBu_r',vlim=(-3,3),
            sensors=True,names=channels,contours=0,image_interp='cubic',extrapolate='head',sphere=sphere,outlines=outlines)
        if n:
            for text in ax.texts:
                if text.get_text() in channels:
                    text.set_color('white' if abs(mean[channels.index(text.get_text()),k])>2 else 'black')
        else:ax.text(.5,.5,'No accepted attempts',ha='center',va='center');ax.axis('off')
        lo,hi=z['band_limits_hz'][k];ax.set_title(f'{bands[k]} ({lo:g}-{hi:g} Hz)')
    fig.colorbar(ScalarMappable(norm=Normalize(-3,3),cmap='RdBu_r'),ax=axes.ravel().tolist(),label='Power change (dB)',extend='both',shrink=.7)
    fig.suptitle(f'Morlet 5 cycles | accepted trials: {n}\nBaseline: preceding rest +6–10s | task: MI +1–5s | mean trial dB')
    fig.supxlabel('Measured electrode values with scalp interpolation; negative = decreased power; descriptive only',fontsize=10)
    fig.savefig(a.output/'topography.png',dpi=150);fig.savefig(a.output/'topography.svg');plt.close(fig)
    write_json(a.output/'plot_manifest.json',dict(kind='scalp band ERD',source_morlet=str(a.morlet.resolve()),
        montage=a.montage,channels=channels,accepted_attempts=n,formula=record['formulas']['topography'],
        sphere_m=sphere.tolist(),display_position_scale=float(scale),vlim_db=[-3,3],no_inference=True))
    print(a.output/'topography.png')


if __name__=='__main__':main()
