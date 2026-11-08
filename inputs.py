"""Dataset discovery is deliberately the same as test(1).py: z1..z4 per nested folder."""
from pathlib import Path
import numpy as np
from PIL import Image
from .config import N

EXTENSIONS={'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.npy'}

def read_intensity(path):
    path=Path(path)
    if path.suffix.lower()=='.npy':
        a=np.load(path)
    else:
        with Image.open(path) as image:
            if image.mode in ('RGB','RGBA','P','LA'):
                image=image.convert('L')
            a=np.asarray(image,dtype=np.float32)
    a=np.asarray(a,dtype=np.float32)
    if a.ndim!=2 or a.shape!=(N,N):
        raise ValueError(f'Expected grayscale {N}x{N}: {path}, got {a.shape}')
    if not np.isfinite(a).all(): raise ValueError(f'Nonfinite image: {path}')
    if (a<0).any(): raise ValueError(f'Negative intensity is not valid: {path}')
    if not (a>0).any(): raise ValueError(f'Completely black image: {path}')
    # IMPORTANT: 8-bit source values left untouched. NO normalizations.
    return a


def read_four(paths):
    if len(paths)!=4: raise ValueError('Exactly four planes are required')
    return np.stack([read_intensity(p) for p in paths],axis=0)


def find_four(directory):
    directory=Path(directory)
    if not directory.is_dir(): raise FileNotFoundError(f'Folder not found: {directory}')
    files=[]
    for k in range(1,5):
        hits=[f for f in directory.iterdir() if f.is_file() and f.stem.lower()==f'z{k}' and f.suffix.lower() in EXTENSIONS]
        if len(hits)!=1: raise ValueError(f'Expected exactly one z{k} image in {directory}; found {[p.name for p in hits]}')
        files.append(hits[0])
    return files


def discover_dataset(root):
    root=Path(root)
    if not root.is_dir(): raise FileNotFoundError(f'Folder not found: {root}')
    # Same search convention as original test(1).py; support root itself as a case.
    dirs=[root]+sorted((x for x in root.rglob('*') if x.is_dir()),key=lambda p:str(p).lower())
    result=[]
    for d in dirs:
        if any(p.is_file() and p.stem.lower() in ('z1','z2','z3','z4') and p.suffix.lower() in EXTENSIONS for p in d.iterdir()):
            result.append((str(d.relative_to(root)) if d!=root else d.name,find_four(d)))
    if not result: raise RuntimeError(f'No z1..z4 image sets in {root}')
    return result
