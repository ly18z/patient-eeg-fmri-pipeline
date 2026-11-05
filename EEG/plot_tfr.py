"""Plot Morlet TFR: average linear trial-baseline ratios, then convert to dB."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from erd import load_results
from ica_common import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--morlet',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--channels',nargs='+')
    a=p.parse_args()
    z,record=load_results(a.morlet);channels=z['channels'].tolist()
    chosen=a.channels or [c for c in ['C3','Cz','C4'] if c in channels]
    ix=[channels.index(c) for c in chosen];power=z['display_power'][:,ix];base=z['baseline_frequency_power'][:,ix,:,None]
    n=len(z['attempt_clusters'])
    db=10*np.log10((power/base).mean(axis=0)) if n else np.full(power.shape[1:],np.nan)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'tfr_values.npz',mean_db=db,
        times_s=z['times_s'],frequencies_hz=z['frequencies_hz'],channels=chosen,attempt_clusters=z['attempt_clusters'])
    cols=min(3,len(chosen));rows=int(np.ceil(len(chosen)/cols))
    fig,axes=plt.subplots(rows,cols,figsize=(cols*4.5,rows*3.8),squeeze=False,layout='constrained')
    for j,ax in enumerate(axes.flat):
        if j>=len(chosen):ax.set_visible(False);continue
        image=ax.pcolormesh(z['times_s'],z['frequencies_hz'],np.ma.masked_invalid(db[j]),cmap='RdBu_r',vmin=-3,vmax=3,shading='auto',rasterized=True)
        ax.axvline(0,c='k',ls='--',lw=1);ax.axvline(1,c='gray',ls=':',lw=1)
        for edge in [8,13,20,30]:ax.axhline(edge,c='gray',ls='--',lw=.6)
        ax.set(title=chosen[j],xlabel='Time from software MI marker (s)',ylabel='Hz',xlim=(-5,5),ylim=(6,35))
        if not len(z['attempt_clusters']):ax.text(.5,.5,'No accepted attempts',transform=ax.transAxes,ha='center',va='center')
    fig.colorbar(image,ax=[ax for ax in axes.flat if ax.get_visible()],label='Power change (dB)',extend='both',shrink=.8)
    fig.suptitle(f'Morlet 5 cycles | accepted trials: {len(z["attempt_clusters"])}\nBaseline: preceding rest +6–10s | task: MI +1–5s | mean linear ratios, then dB')
    fig.savefig(a.output/'tfr.png',dpi=150);fig.savefig(a.output/'tfr.svg');plt.close(fig)
    write_json(a.output/'plot_manifest.json',dict(kind='TFR',source_morlet=str(a.morlet.resolve()),
        channels=chosen,vlim_db=[-3,3],accepted_attempts=len(z['attempt_clusters']),formula=record['formulas']['tfr'],
        no_inference=True))
    print(a.output/'tfr.png')


if __name__=='__main__':main()
