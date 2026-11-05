"""Run fMRIPrep on BIDS, with explicit optional template downloads."""
import argparse
import json
from pathlib import Path
import subprocess

PARAMETERS=dict(cpus=8,omp_threads=4,memory='13g',memory_mb=12000,seed=20261026)


def command(config):
    c=config
    args=[c['docker'],'run','--rm','--pull','never','--platform','linux/amd64',
          '--cpus',str(c['cpus']),'--memory',c['memory'],'--shm-size','2g']
    if not c.get('allow_template_downloads',False):args+=['--network','none']
    for src,dst,ro in [(c['bids'],'/data',True),(c['output'],'/out',False),(c['work'],'/work',False),
                       (c['templateflow'],'/templateflow',not c.get('allow_template_downloads',False)),(c['license'],'/license.txt',True)]:
        args+=['--mount',f'type=bind,src={Path(src).resolve()},dst={dst}'+(',readonly' if ro else '')]
    args+=['-e','TEMPLATEFLOW_HOME=/templateflow','-e',
           'TEMPLATEFLOW_AUTOUPDATE='+('1' if c.get('allow_template_downloads',False) else '0'),c['image'],
           '/data','/out','participant','--participant-label',c['participant'],'--fs-no-reconall',
           '--output-spaces',*c['output_spaces'],'--nprocs',str(c['cpus']),'--omp-nthreads',str(c['omp_threads']),
           '--mem',str(c['memory_mb']),'--random-seed',str(c['seed']),'--skull-strip-fixed-seed',
           '--fs-license-file','/license.txt','--notrack','--stop-on-first-crash','-w','/work']
    if c['ignore']:args+=['--ignore',*c['ignore']]
    return args


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bids','output','templateflow','license']:p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--participant',required=True);p.add_argument('--work',type=Path)
    p.add_argument('--docker',default='docker')
    p.add_argument('--image',default='nipreps/fmriprep:25.2.7')
    p.add_argument('--output-spaces',nargs='+',default=['T1w','MNI152NLin2009cAsym:res-2'])
    p.add_argument('--ignore',nargs='*',default=[])
    p.add_argument('--allow-template-downloads',action='store_true')
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    output=a.output;work=a.work or output.with_name(output.name+'_work')
    c=PARAMETERS|vars(a)|{'work':work};args=command(c)
    if a.dry_run:print(json.dumps(args,ensure_ascii=False,indent=2));return
    protected=[Path(c[k]).resolve() for k in ['bids','templateflow']]
    for target in [output.resolve(),work.resolve()]:
        if any(target.is_relative_to(x) or x.is_relative_to(target) for x in protected):
            raise ValueError('Output/work must not overlap BIDS or template caches')
    if output.resolve().is_relative_to(work.resolve()) or work.resolve().is_relative_to(output.resolve()):
        raise ValueError('Output and work must be separate')
    output.mkdir(parents=True);work.mkdir(parents=True)
    if a.allow_template_downloads:a.templateflow.mkdir(parents=True,exist_ok=True)
    image_id=subprocess.check_output([c['docker'],'image','inspect','--format','{{.Id}}',c['image']],text=True).strip()
    record=dict(status='RUNNING',image_id=image_id,command=args,configuration=c)
    manifest=output/'pipeline_execution.json'
    manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    with (output/'fmriprep.log').open('w',encoding='utf-8') as log:r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
    record.update(status='PROCESS_FINISHED_QC_PENDING' if r.returncode==0 else 'FAILED',exit_code=r.returncode)
    manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    if r.returncode:raise SystemExit(r.returncode)
    print('fMRIPrep finished; numerical and visual QC are still required.')


if __name__=='__main__':main()
