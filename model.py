"""Original trained CNN + Fourier MLP + TT model from validated ablation notebook.

Module/parameter names preserved so saved trained checkpoint loads with strict=True.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import MAX_PHASE_STEP_RAD

class ConvBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),
                               nn.GroupNorm(8,channels),nn.SiLU(),
                               nn.Conv2d(channels,channels,3,padding=1),
                               nn.GroupNorm(8,channels))
    def forward(self,x): return F.silu(x+self.net(x))

class PhysicalTensorFourier(nn.Module):
    """Same supernetwork initial weights; disabled branches feed literal zeros."""
    def __init__(self,n,rank,basis,variant='full',channels=32):
        super().__init__()
        assert variant in ('cnn','cnn_fourier','full')
        self.variant=variant; self.n=n; self.rank=rank
        self.downsample=1
        self.use_fourier=variant in ('cnn_fourier','full')
        self.use_tt=variant=='full'
        self.encoder=nn.Sequential(nn.Conv2d(17,channels,3,padding=1),
                                   nn.GroupNorm(8,channels),nn.SiLU(),
                                   ConvBlock(channels),ConvBlock(channels))
        self.global_embed=nn.Sequential(nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),
                                        nn.Linear(channels*16,128),nn.SiLU())
        self.register_buffer('basis',basis.detach().clone(),persistent=False)
        K=basis.shape[0]
        self.fourier_mlp=nn.Sequential(nn.Linear(128+2*K,192),nn.SiLU(),
                                       nn.Linear(192,192),nn.SiLU(),nn.Linear(192,K))
        nn.init.zeros_(self.fourier_mlp[-1].weight)
        nn.init.zeros_(self.fourier_mlp[-1].bias)
        logn=round(math.log2(n)); assert n==2**logn, 'TT requires power-of-two image size'
        a=2**(logn//2); b=n//a
        self.core_shapes=[(1,a,rank),(rank,b,rank),(rank,a,rank),(rank,b,1)]
        self.tt_cores=nn.ParameterList()
        self.tt_heads=nn.ModuleList()
        for shape in self.core_shapes:
            self.tt_cores.append(nn.Parameter(0.12*torch.randn(shape)))
            head=nn.Linear(128,math.prod(shape))
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
            self.tt_heads.append(head)
        # IMPORTANT: exact same decoder input channels / shapes in ALL variants.
        self.decoder=nn.Sequential(nn.Conv2d(channels+3,channels,3,padding=1),
                                   nn.GroupNorm(8,channels),nn.SiLU(),
                                   ConvBlock(channels),nn.Conv2d(channels,1,3,padding=1))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        # The inactive blocks do not consume optimizer budget/weight decay.
        if not self.use_fourier:
            for p in self.fourier_mlp.parameters(): p.requires_grad_(False)
        if not self.use_tt:
            for p in self.tt_cores: p.requires_grad_(False)
            for p in self.tt_heads.parameters(): p.requires_grad_(False)
        if not (self.use_tt or self.use_fourier):
            for p in self.global_embed.parameters(): p.requires_grad_(False)

    def make_tt(self,code):
        cores=[]
        for base,head,shape in zip(self.tt_cores,self.tt_heads,self.core_shapes):
            cores.append(base+0.15*torch.tanh(head(code).reshape(shape)))
        A,B,C,D=cores
        return torch.einsum('ia,ajb,bkc,cl->ijkl',A[0],B,C,D[:,:,0]).reshape(self.n,self.n)

    def forward(self,features,g,phase):
        # Optional 2x CNN spatial reduction. The Fourier coefficients and TT
        # representation are STILL computed at original 128x128 resolution.
        scale=self.downsample
        feature_low=F.avg_pool2d(features,scale) if scale>1 else features
        h=self.encoder(feature_low)
        zero=torch.zeros_like(g)
        code=self.global_embed(h)[0] if (self.use_fourier or self.use_tt) else None
        if self.use_fourier:
            spec_g=torch.einsum('khw,hw->k',self.basis,g)/(self.n*self.n)
            spec_phi=torch.einsum('khw,hw->k',self.basis,phase)/(self.n*self.n)
            coefficients=0.2*torch.tanh(self.fourier_mlp(torch.cat([code,spec_g,spec_phi])))
            spectral=torch.einsum('k,khw->hw',coefficients,self.basis)/math.sqrt(len(coefficients))
        else:
            spectral=zero
        tensor_map=torch.tanh(self.make_tt(code)) if self.use_tt else zero
        if scale>1:
            # Whole-network weights remain checkpoint-compatible.
            g_low=F.avg_pool2d(g[None,None],scale)
            spectral_low=F.avg_pool2d(spectral[None,None],scale)
            tt_low=F.avg_pool2d(tensor_map[None,None],scale)
        else:
            g_low=g[None,None]
            spectral_low=spectral[None,None]
            tt_low=tensor_map[None,None]
        fused=self.decoder(torch.cat([h,g_low,spectral_low,tt_low],dim=1))
        if scale>1:
            fused=F.interpolate(fused,size=(self.n,self.n),mode='bilinear',align_corners=False)
        correction=MAX_PHASE_STEP_RAD*torch.tanh(fused[0,0])
        return correction-correction.mean()
