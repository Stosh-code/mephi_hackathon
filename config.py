from dataclasses import dataclass

N = 128
DX = 10e-6
WAVELENGTH = 532e-9
Z_PLANES = (0.0241, 0.0253, 0.0265, 0.0277)
PAD_FACTOR = 4
INCIDENT_AMPLITUDE = 1.0
SSIM_DATA_RANGE = 255.0
FOURIER_MODES = 64
TT_RANK = 6
MAX_PHASE_STEP_RAD = 0.2
BASE_GRADIENT_STEP_RAD = 0.12
BACKTRACK_FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.0)

@dataclass
class RunConfig:
    steps: int = 100
    inner_steps: int = 1
    nn_downsample: int = 2
    learning_rate: float = 3e-4
    last_qat_steps: int = 0
    grad_clip: float = 1.0
    seed: int = 20261108
    propagator: str = "fresnel"
    ssim_weight: float = 0.60
    pearson_weight: float = 0.40
    # No loss term that silently matches the intensity scales.
    nrmse_weight: float = 0.0
    device: str = "cpu"
    threads: int = 4
    verbose_every: int = 10

    def validate(self):
        if self.nn_downsample not in (1,2): raise ValueError("nn_downsample must be 1 or 2")
        if self.steps < 0 or self.steps > 100:
            raise ValueError("steps must be between 0 and 100")
        if self.inner_steps < 1 or self.inner_steps > 10:
            raise ValueError("inner_steps must be between 1 and 10")
        if self.propagator not in ("fresnel", "asm"):
            raise ValueError("propagator must be fresnel or asm")
        if min(self.ssim_weight, self.pearson_weight, self.nrmse_weight) < 0:
            raise ValueError("loss weights must be nonnegative")
