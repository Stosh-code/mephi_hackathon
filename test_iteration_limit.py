from pathlib import Path
from doe.config import RunConfig


def test_fixed_step_config():
    cfg = RunConfig()
    assert cfg.steps == 100
    assert not hasattr(cfg, 'time_limit_s')
    cfg.validate()
    for value in (0, 1, 100):
        RunConfig(steps=value).validate()
    try:
        RunConfig(steps=101).validate()
    except ValueError:
        pass
    else:
        raise AssertionError('101 steps must be forbidden')

if __name__ == '__main__':
    test_fixed_step_config()
    print('Fixed-step 100 limit OK')
