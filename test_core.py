import math
import numpy as np
import torch
from doe.config import N,DX,WAVELENGTH,Z_PLANES
from doe.optics import ASM,FresnelOneFFT,phase_to_bytes,bytes_to_phase
from doe.tie import derivative_4point
from doe.optimizer import make_model


def test_phase_encoding():
    p=torch.tensor([-math.pi,0.0,math.pi],dtype=torch.float32)
    code=phase_to_bytes(p)
    assert code.tolist()==[0,128,0]
    decoded=bytes_to_phase(torch.tensor([0,255],dtype=torch.uint8))
    assert abs(float(decoded[0])+math.pi)<1e-6
    assert abs(float(decoded[1])-math.pi)<1e-6


def test_four_point_derivative():
    z=np.array([-1.5,-0.5,0.5,1.5]); y=z**3+2*z
    arr=np.stack([np.full((3,3),val) for val in y])
    deriv=derivative_4point(arr,1.)
    assert np.max(np.abs(deriv-2))<1e-12


def test_fresnel_power():
    p=FresnelOneFFT()
    field=torch.ones(N,N,dtype=torch.complex64)
    out=p(field)
    for j,z in enumerate(Z_PLANES):
        dx_out=WAVELENGTH*z/(N*DX)
        input_energy=float((field.abs().square().sum()*DX**2))
        output_energy=float((out[j].abs().square().sum()*dx_out**2))
        assert abs(output_energy/input_energy-1)<1e-4


def test_asm_identity():
    p=ASM(z_list=(0.,))
    f=torch.exp(1j*torch.rand(N,N))
    assert torch.max((p(f)[0]-f).abs())<1e-5


def test_random_model_build():
    model=make_model(None,'cpu')
    assert sum(x.numel() for x in model.parameters() if x.requires_grad)==366785


def test_full_model_downsampling():
    torch.set_num_threads(2)
    for scale in (1,2):
        model=make_model(None,'cpu',nn_downsample=scale)
        x=torch.zeros(1,17,N,N)
        g=torch.ones(N,N)
        phase=torch.zeros(N,N)
        y=model(x,g,phase)
        assert y.shape==(N,N)
        assert torch.isfinite(y).all()
