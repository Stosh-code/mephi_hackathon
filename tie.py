"""Regularized finite-volume TIE initializer using all four target intensity planes."""
import math
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator
from scipy.ndimage import gaussian_filter
import torch
from .config import DX, WAVELENGTH, Z_PLANES
from .optics import ASM


def derivative_4point(images,h):
    if len(images)!=4 or h<=0: raise ValueError('TIE4 requires four equally spaced planes')
    return (images[0]-27*images[1]+27*images[2]-images[3])/(24*h)


def midpoint_4point(images):
    if len(images)!=4: raise ValueError('TIE4 requires four planes')
    return (-images[0]+9*images[1]+9*images[2]-images[3])/16


def build_tie_matrix(I,dx,reg_fraction=4e-3):
    n,m=I.shape
    if n!=m: raise ValueError('square intensity required')
    I=np.maximum(I,1e-4).astype(np.float64)
    wx=2*I[:,:-1]*I[:,1:]/(I[:,:-1]+I[:,1:]+1e-12)/dx**2
    wy=2*I[:-1,:]*I[1:,:]/(I[:-1,:]+I[1:,:]+1e-12)/dx**2
    pos=np.arange(n*n).reshape(n,n)
    diag=np.full(n*n,reg_fraction*np.mean(I)/dx**2,dtype=np.float64)
    rows=[];cols=[];vals=[]
    for a,b,w in ((pos[:,:-1],pos[:,1:],wx),(pos[:-1,:],pos[1:,:],wy)):
        ia=a.ravel();ib=b.ravel();wij=w.ravel()
        diag[ia]+=wij;diag[ib]+=wij
        rows.extend([ia,ib]);cols.extend([ib,ia]);vals.extend([-wij,-wij])
    rows.append(np.arange(n*n));cols.append(np.arange(n*n));vals.append(diag)
    rows=np.concatenate(rows);cols=np.concatenate(cols);vals=np.concatenate(vals)
    return sparse.coo_matrix((vals,(rows,cols)),shape=(n*n,n*n)).tocsr()


def solve_tie_phase(Ic,dIdz,dx=DX,wavelength=WAVELENGTH,reg=4e-3):
    k=2*math.pi/wavelength
    A=build_tie_matrix(Ic,dx,reg)
    rhs=k*(dIdz-dIdz.mean()).ravel()
    diagonal=A.diagonal()
    pre=LinearOperator(A.shape,matvec=lambda x:x/diagonal)
    phi,info=cg(A,rhs,M=pre,rtol=2e-5,atol=0,maxiter=1500)
    if info!=0: warnings.warn(f'TIE CG not fully converged, info={info}')
    phi=phi.reshape(Ic.shape)
    phi-=phi.mean()
    residual=float(np.linalg.norm(A@phi.ravel()-rhs)/(np.linalg.norm(rhs)+1e-20))
    return phi,residual


def initialize_tie4(targets_np,device):
    """Return phase-only screen; no normalization of target images.

    Original amplitude sqrt(Ic) is used only in approximate TIE backward pass;
    resulting DOE is phase-only, with unit incident amplitude in forward model.
    """
    h=Z_PLANES[1]-Z_PLANES[0]
    assert np.allclose(np.diff(Z_PLANES),h,rtol=1e-7,atol=1e-12)
    im=np.stack([gaussian_filter(x.astype(np.float64),1.0,mode='constant',cval=0) for x in targets_np])
    derivative=derivative_4point(im,h)
    Ic_raw=midpoint_4point(im)
    floor=max(1e-9,1e-3*max(float(Ic_raw.max()),0.0))
    Ic=np.maximum(Ic_raw,floor)
    phi,residual=solve_tie_phase(Ic,derivative)
    zc=0.5*(Z_PLANES[1]+Z_PLANES[2])
    with torch.no_grad():
        field=torch.tensor(np.sqrt(Ic)*np.exp(1j*phi),device=device,dtype=torch.complex64)
        backwards=ASM(z_list=(-zc,)).to(device)
        screen=torch.angle(backwards(field)[0]).float().detach()
    if not torch.isfinite(screen).all(): raise ValueError('TIE phase has nonfinite values')
    return screen,{'pde_residual':residual,'negative_midpoint_fraction':float((Ic_raw<=0).mean())}
