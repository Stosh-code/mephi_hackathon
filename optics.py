"""Differentiable Fresnel one-FFT (competition statement) and padded ASM (research baseline)."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import N, DX, WAVELENGTH, Z_PLANES, PAD_FACTOR, INCIDENT_AMPLITUDE


def phase_to_bytes(phase):
    """uint8 0 -> -pi, 255 -> +pi; periodic endpoints equivalent physically."""
    wrapped = torch.remainder(phase + math.pi, 2*math.pi)
    values = torch.round(wrapped * (255.0/(2*math.pi))).clamp(0,255)
    return values.to(torch.uint8)


def bytes_to_phase(code):
    return code.to(torch.float32)*(2*math.pi/255.0)-math.pi


def phase_8bit_ste(phase):
    phase_q = bytes_to_phase(phase_to_bytes(phase)).to(phase.device)
    return phase + (phase_q - phase).detach()


class ASM(nn.Module):
    def __init__(self, n=N, dx=DX, wavelength=WAVELENGTH, z_list=Z_PLANES,
                 pad_factor=PAD_FACTOR, bandlimit=True):
        super().__init__()
        self.n=int(n); self.p=int(pad_factor*n); self.a=(self.p-self.n)//2
        self.asm_calls=0
        self.z_list=tuple(float(z) for z in z_list)
        f=torch.fft.fftfreq(self.p,d=dx,dtype=torch.float64)
        FY,FX=torch.meshgrid(f,f,indexing='ij')
        k=2*math.pi/wavelength
        q2=k*k-(2*math.pi*FX)**2-(2*math.pi*FY)**2
        kz=torch.sqrt(q2.clamp_min(0.0)); alpha=torch.sqrt((-q2).clamp_min(0.0))
        propagating=(q2>=0)
        H=[]
        for z in self.z_list:
            if z==0:
                transfer=torch.ones_like(kz,dtype=torch.complex128)
            else:
                transfer=torch.exp(1j*z*(kz-k))
                if z>0: transfer=transfer*torch.exp(-z*alpha)
                else: transfer=transfer*propagating.to(transfer.dtype)
                if bandlimit:
                    L=self.p*dx
                    fcut=1/(wavelength*math.sqrt(1+(2*abs(z)/L)**2))
                    transfer=transfer*((FX.abs()<=fcut)&(FY.abs()<=fcut))
            H.append(transfer.to(torch.complex64))
        self.register_buffer('transfer',torch.stack(H),persistent=False)

    def forward(self,field):
        self.asm_calls+=1
        assert field.shape==(self.n,self.n)
        left=self.a;right=self.p-self.n-left
        padded=F.pad(field,(left,right,left,right))
        spec=torch.fft.fft2(padded,norm='ortho')
        full=torch.fft.ifft2(spec[None]*self.transfer,norm='ortho')
        return full[:,left:left+self.n,left:left+self.n]

    def intensities(self,phase,quantized=False):
        p=phase_8bit_ste(phase) if quantized else phase
        field=INCIDENT_AMPLITUDE*torch.exp(1j*p)
        return self(field).abs().square()


class FresnelOneFFT(nn.Module):
    """Direct sampled Fresnel integral with a single centered FFT.

    Output sampling for each z: dx_out = wavelength*z/(n*dx).
    No target-dependent amplitude scaling and no wraparound due to field padding.
    """
    def __init__(self, n=N, dx=DX, wavelength=WAVELENGTH, z_list=Z_PLANES):
        super().__init__()
        self.n=n; self.z_list=tuple(z_list); self.calls=0
        x=(torch.arange(n,dtype=torch.float64)-n//2)*dx
        yy,xx=torch.meshgrid(x,x,indexing='ij')
        r2_in=xx*xx+yy*yy
        input_chirps=[]; output_chirps=[]; amplitudes=[]
        for z in self.z_list:
            if z==0: raise ValueError('single FFT Fresnel requires z != 0')
            pitch_out=wavelength*abs(z)/(n*dx)
            out=(torch.arange(n,dtype=torch.float64)-n//2)*pitch_out
            yo,xo=torch.meshgrid(out,out,indexing='ij')
            input_chirps.append(torch.exp(1j*math.pi*r2_in/(wavelength*z)).to(torch.complex64))
            output_chirps.append(torch.exp(1j*math.pi*(xo*xo+yo*yo)/(wavelength*z)).to(torch.complex64))
            # norm='ortho': FFT2 is 1/n times the unnormalized FFT2
            amplitudes.append(n*dx*dx/(wavelength*abs(z)))
        self.register_buffer('chirp_in',torch.stack(input_chirps),persistent=False)
        self.register_buffer('chirp_out',torch.stack(output_chirps),persistent=False)
        self.register_buffer('amp',torch.tensor(amplitudes,dtype=torch.float32),persistent=False)

    def forward(self,field):
        self.calls+=1
        assert field.shape==(self.n,self.n)
        weighted=field[None]*self.chirp_in
        transformed=torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(weighted,dim=(-2,-1)),norm='ortho'),dim=(-2,-1))
        return transformed*self.chirp_out*self.amp[:,None,None]

    def intensities(self,phase,quantized=False):
        p=phase_8bit_ste(phase) if quantized else phase
        return self(INCIDENT_AMPLITUDE*torch.exp(1j*p)).abs().square()


def build_propagator(kind,device):
    return (ASM() if kind=='asm' else FresnelOneFFT()).to(device)
