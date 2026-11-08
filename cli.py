"""Docker entry point. One set of 4 images -> DOE PNG + 4 reconstructions + metrics."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .config import RunConfig
from .inputs import read_four,find_four,discover_dataset
from .optics import build_propagator
from .optimizer import synthesize


def write_results(files,output_png,prop,cfg):
    print('\nInput:',', '.join(str(p) for p in files),flush=True)
    target=read_four(files)
    codes,reconstructed,report,history,_state=synthesize(target,prop,cfg,None)
    output_png=Path(output_png)
    output_png.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(codes,mode='L').save(output_png)
    with Image.open(output_png) as check:
        if check.mode!='L' or check.size!=(128,128):
            raise RuntimeError('Output PNG validation failed')
    side=output_png.parent
    with (side/(output_png.stem+'_metrics.json')).open('w',encoding='utf-8') as fp:
        json.dump(report,fp,indent=2,ensure_ascii=False)
    with (side/(output_png.stem+'_history.csv')).open('w',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=['step','accepted','P','SSIM','loss','elapsed_s'])
        writer.writeheader();writer.writerows(history)
    for j,img in enumerate(reconstructed,1):
        display=np.clip(np.round(img),0,255).astype(np.uint8)
        Image.fromarray(display,mode='L').save(side/(output_png.stem+f'_reconstruction_z{j}.png'))
    print(f"Wrote {output_png} | Pearson={report['best']['mean_P']:.4f} "
          f"SSIM={report['best']['mean_SSIM']:.4f} | steps={report['steps_completed']}"
          f" | algorithm_time={report['time_s']:.2f}s",flush=True)
    print('Timing breakdown (s):',report['timing_s'],flush=True)
    return {'output':str(output_png),'P':report['best']['mean_P'],
            'SSIM':report['best']['mean_SSIM'],'time_s':report['time_s']}


def main(argv=None):
    parser=argparse.ArgumentParser(description='TIE4 + physics-guided CNN-Fourier-TT, 4 images -> DOE + reconstructions + metrics')
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input-dir',type=Path,help='Folder containing z1.png...z4.png')
    source.add_argument('--input-root',type=Path,help='Root containing nested folders of z1..z4')
    source.add_argument('--images',nargs=4,type=Path,metavar=('Z1','Z2','Z3','Z4'),help='Four images in propagation order')
    parser.add_argument('--output',type=Path,default=Path('/output/doe.png'),help='Output PNG path for one case; directory for --input-root')
    parser.add_argument('--steps',type=int,default=100,help='Per-case optimizer iterations, 0..100')
    parser.add_argument('--qat-last-steps',type=int,default=0,help='Experimental quantization-aware last steps (default OFF)')
    parser.add_argument('--inner-steps',type=int,default=1)
    parser.add_argument('--nn-downsample',type=int,choices=[1,2],default=2,help='CNN resolution reduction (2 = faster, TT and Fourier stay full-res)')
    parser.add_argument('--propagator',choices=['fresnel','asm'],default='fresnel',help='Fresnel one FFT matches PDF model; ASM is research baseline')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--seed',type=int,default=20261108)
    parser.add_argument('--ssim-weight',type=float,default=0.60)
    parser.add_argument('--pearson-weight',type=float,default=0.40)
    parser.add_argument('--nrmse-weight',type=float,default=0.0)
    a=parser.parse_args(argv)
    import torch
    dev=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    cfg=RunConfig(steps=a.steps,inner_steps=a.inner_steps,nn_downsample=a.nn_downsample,last_qat_steps=a.qat_last_steps,propagator=a.propagator,
                  device=dev,threads=a.threads,seed=a.seed,
                  ssim_weight=a.ssim_weight,pearson_weight=a.pearson_weight,nrmse_weight=a.nrmse_weight)
    cfg.validate()
    prop=build_propagator(a.propagator,dev)
    print(f'Propagator={a.propagator}, device={dev}, initialization=random_per_case',flush=True)
    if a.images:
        entries=[('single',a.images)]
    elif a.input_dir:
        entries=[(a.input_dir.name,find_four(a.input_dir))]
    else:
        entries=discover_dataset(a.input_root)
    if len(entries)==1 and not a.input_root:
        out=a.output if a.output.suffix.lower()=='.png' else a.output/'doe.png'
        write_results(entries[0][1],out,prop,cfg)
    else:
        a.output.mkdir(parents=True,exist_ok=True)
        summary=[]
        for i,(name,files) in enumerate(entries,1):
            print(f'\nCASE {i}/{len(entries)}: {name}',flush=True)
            case_dir=a.output/name
            case_cfg=RunConfig(**{**vars(cfg),'seed':cfg.seed+i-1})
            summary.append(write_results(files,case_dir/'doe.png',prop,case_cfg))
        with (a.output/'summary.csv').open('w',newline='',encoding='utf-8') as fp:
            writer=csv.DictWriter(fp,fieldnames=['output','P','SSIM','time_s'])
            writer.writeheader();writer.writerows(summary)
        print(f'Completed {len(summary)} cases. Summary: {a.output/"summary.csv"}',flush=True)


if __name__=='__main__':main()
