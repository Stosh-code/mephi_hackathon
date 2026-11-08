"""Per-plane Pearson and SSIM plus RMSE; never normalizes target or prediction."""
import torch
from torch.nn import functional as F
from .config import SSIM_DATA_RANGE


def per_plane_metrics(pred,target):
    if pred.shape!=target.shape or pred.ndim!=3 or pred.shape[0]!=4:
        raise ValueError('Need four predicted and four target intensity matrices')
    X=pred.flatten(1); Y=target.flatten(1)
    xc=X-X.mean(dim=1,keepdim=True);yc=Y-Y.mean(dim=1,keepdim=True)
    pearson=(xc*yc).sum(dim=1)/(xc.norm(dim=1)*yc.norm(dim=1)+1e-8)
    rmse=(pred-target).square().mean(dim=(-1,-2)).sqrt()
    nrmse=rmse/(target.square().mean(dim=(-1,-2)).sqrt()+1e-8)
    coord=torch.arange(-3,4,device=target.device,dtype=target.dtype)
    gaussian=torch.exp(-coord.square()/(2*1.5**2));gaussian/=gaussian.sum()
    window=(gaussian[:,None]*gaussian[None,:])[None,None]
    def blur(x):return F.conv2d(F.pad(x,(3,3,3,3),mode='reflect'),window)
    a=pred[:,None];b=target[:,None]
    ma=blur(a);mb=blur(b)
    va=(blur(a*a)-ma*ma).clamp_min(0.0)
    vb=(blur(b*b)-mb*mb).clamp_min(0.0)
    cov=blur(a*b)-ma*mb
    c1=(0.01*SSIM_DATA_RANGE)**2;c2=(0.03*SSIM_DATA_RANGE)**2
    ssim=((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2)+1e-12)
    return dict(P=pearson,SSIM=ssim.mean(dim=(1,2,3)),RMSE=rmse,NRMSE=nrmse)


def metrics_and_loss(pred,target,cfg):
    metrics=per_plane_metrics(pred,target)
    loss=(cfg.pearson_weight*(1-metrics['P'])+
          cfg.ssim_weight*(1-metrics['SSIM'])+
          cfg.nrmse_weight*torch.log1p(metrics['NRMSE'])).mean()
    return loss,metrics


def evaluate(phase,target,prop,cfg,quantized=True):
    from .optics import phase_to_bytes,bytes_to_phase
    with torch.no_grad():
        # Quantized phase is *exactly* the values that will be written to PNG.
        phi=bytes_to_phase(phase_to_bytes(phase)).to(phase.device) if quantized else phase
        I=prop.intensities(phi,quantized=False)
        loss,values=metrics_and_loss(I,target,cfg)
    data={k:[float(v) for v in tensor.cpu()] for k,tensor in values.items()}
    data.update({f'mean_{k}':float(sum(v)/4) for k,v in data.items()})
    data['loss']=float(loss)
    data['Qmin']=min((p+s)/2 for p,s in zip(data['P'],data['SSIM']))
    return data,I.detach()
