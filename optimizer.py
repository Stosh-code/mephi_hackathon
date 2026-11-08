"""Sequential gradient-based phase refinement, full CNN+Fourier+TT, per-case fine-tuning."""
import copy
import math
import time
import numpy as np
import torch
from .config import N,FOURIER_MODES,TT_RANK,MAX_PHASE_STEP_RAD,BASE_GRADIENT_STEP_RAD,BACKTRACK_FACTORS
from .model import PhysicalTensorFourier
from .metrics import metrics_and_loss, evaluate
from .tie import initialize_tie4
from .optics import phase_to_bytes


def make_fourier_basis(n,modes,device):
    grid=torch.arange(n,device=device,dtype=torch.float32)/n
    yy,xx=torch.meshgrid(grid,grid,indexing='ij')
    pairs=[];bound=max(5,int(math.ceil(math.sqrt(modes))))
    for kx in range(-bound,bound+1):
        for ky in range(-bound,bound+1):
            if kx>0 or (kx==0 and ky>0): pairs.append((kx,ky))
    pairs.sort(key=lambda a:(a[0]**2+a[1]**2,a[0],a[1]))
    func=[]
    for kx,ky in pairs:
        omega=2*math.pi*(kx*xx+ky*yy)
        func.extend([math.sqrt(2)*torch.cos(omega),math.sqrt(2)*torch.sin(omega)])
        if len(func)>=modes:break
    basis=torch.stack(func[:modes],dim=0)
    return basis


def physics_gradient(phase,target,prop,cfg,quantized=False):
    with torch.enable_grad():
        probe=phase.detach().clone().requires_grad_(True)
        pred=prop.intensities(probe,quantized=quantized)
        loss,_=metrics_and_loss(pred,target,cfg)
        grad=torch.autograd.grad(loss,probe,create_graph=False,retain_graph=False)[0]
    rms=grad.square().mean().sqrt().clamp_min(1e-12)
    return (grad/rms).detach(),pred.detach(),float(loss.detach())


def network_features(phase,target,pred,g,coords):
    residual=target-pred
    return torch.cat([torch.cos(phase)[None,None],torch.sin(phase)[None,None],
                      target[None],pred[None],residual[None],g[None,None],coords],dim=1)


def choose_step(current,delta,target,prop,cfg,quantized,base_score):
    with torch.no_grad():
        score=base_score
        for alpha in BACKTRACK_FACTORS:
            if alpha==0:continue
            trial=current.detach()+alpha*delta.detach()
            trial_loss,_=metrics_and_loss(prop.intensities(trial,quantized),target,cfg)
            if float(trial_loss)<score-1e-7: return alpha
    return 0.0


def iterate_phase(phase,target,model,prop,cfg,coords,steps,qat):
    losses=[];accepted=[]
    for _ in range(steps):
        g,pred,score=physics_gradient(phase,target,prop,cfg,quantized=qat)
        feat=network_features(phase,target,pred,g,coords)
        net_delta=model(feat,g,phase)
        delta=-BASE_GRADIENT_STEP_RAD*g+net_delta
        alpha=choose_step(phase,delta,target,prop,cfg,qat,score)
        phase=phase+alpha*delta
        loss,_=metrics_and_loss(prop.intensities(phase,quantized=qat),target,cfg)
        losses.append(loss);accepted.append(alpha)
    weights=torch.arange(1,len(losses)+1,device=phase.device,dtype=phase.dtype)
    weights=weights/weights.sum()
    total=sum(w*loss for w,loss in zip(weights,losses))
    return total,phase,accepted


def make_model(checkpoint_path,device,nn_downsample=1):
    # Same network/parameter names as original full model, checkpoint loads strictly.
    basis=make_fourier_basis(N,FOURIER_MODES,device)
    model=PhysicalTensorFourier(N,TT_RANK,basis,variant='full').to(device)
    if checkpoint_path:
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        state=checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state,strict=True)
    model.downsample=nn_downsample
    return model


def synthesize(images,prop,cfg,checkpoint_path=None):
    cfg.validate()
    np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    if cfg.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    device=torch.device(cfg.device)
    torch.set_num_threads(cfg.threads)
    target=torch.from_numpy(images.copy()).to(device=device,dtype=torch.float32)
    def clock():
        if device.type=='cuda': torch.cuda.synchronize(device)
        return time.perf_counter()
    start=clock()
    tie,tie_info=initialize_tie4(images,device)
    t_after_tie=clock()
    base, _=evaluate(tie,target,prop,cfg,quantized=True)
    t_after_eval=clock()
    model=make_model(checkpoint_path,device,nn_downsample=cfg.nn_downsample)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=1e-5)
    t_after_setup=clock()
    coords_v=torch.linspace(-1,1,N,device=device,dtype=torch.float32)
    yy,xx=torch.meshgrid(coords_v,coords_v,indexing='ij')
    coords=torch.stack([xx,yy])[None]
    phase=tie.detach().clone()
    best_phase=phase.clone();best_metric=base
    best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    history=[]
    completed_steps=0
    step_seconds=[]
    if cfg.steps==0:
        model.eval()
        _,proposed,accepted=iterate_phase(phase,target,model,prop,cfg,coords,cfg.inner_steps,False)
        metric,_=evaluate(proposed,target,prop,cfg,quantized=True)
        if metric['loss']<best_metric['loss']:
            best_phase=proposed.detach();best_metric=metric
        history.append({'step':0,'accepted':int(sum(a>0 for a in accepted)),
                        'P':best_metric['mean_P'],'SSIM':best_metric['mean_SSIM'],
                        'loss':best_metric['loss'],'elapsed_s':clock()-start})
    for epoch in range(1,cfg.steps+1):
        step_start=clock()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        qat=cfg.last_qat_steps>0 and epoch>cfg.steps-cfg.last_qat_steps
        train_loss,proposed,accepted=iterate_phase(phase.detach(),target,model,prop,cfg,coords,cfg.inner_steps,qat)
        if not torch.isfinite(train_loss):raise FloatingPointError(f'Nonfinite training loss at {epoch}')
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
        optimizer.step()
        completed_steps=epoch
        step_seconds.append(clock()-step_start)
        phase=proposed.detach()
        # Snapshot by *actual exported 8-bit PNG* score, not continuous phase.
        if epoch==1 or epoch%cfg.verbose_every==0 or epoch==cfg.steps:
            metric,_=evaluate(phase,target,prop,cfg,quantized=True)
            if metric['loss']<best_metric['loss']:
                best_phase=phase.detach().clone();best_metric=metric
                best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            row={'step':epoch,'accepted':sum(a>0 for a in accepted),
                 'P':metric['mean_P'],'SSIM':metric['mean_SSIM'],
                 'loss':metric['loss'],'elapsed_s':clock()-start}
            history.append(row)
            print(f"  {epoch:3d}/{cfg.steps}: P={row['P']:.4f} SSIM={row['SSIM']:.4f} "
                  f"loss={row['loss']:.4f} accepted={row['accepted']}/{cfg.inner_steps} "
                  f"t={row['elapsed_s']:.1f}s",flush=True)
    t_after_training=clock()
    model.load_state_dict(best_weights)
    best_metric,final_intensities=evaluate(best_phase,target,prop,cfg,quantized=True)
    t_after_final=clock()
    result={'baseline':base,'best':best_metric,'tie_info':tie_info,
            'steps_requested':cfg.steps,'steps_completed':completed_steps,
            'max_steps':100,'nn_downsample':cfg.nn_downsample,
            'propagator':cfg.propagator,
            'time_s':t_after_final-start,
            'timing_s':{'tie':t_after_tie-start,'initial_evaluation':t_after_eval-t_after_tie,
                        'model_setup':t_after_setup-t_after_eval,
                        'optimization':t_after_training-t_after_setup,
                        'final_evaluation':t_after_final-t_after_training},
            'mean_step_ms':1000*sum(step_seconds)/max(1,len(step_seconds))}
    return (phase_to_bytes(best_phase).detach().cpu().numpy(),
            final_intensities.detach().cpu().numpy(),result,history,best_weights)
