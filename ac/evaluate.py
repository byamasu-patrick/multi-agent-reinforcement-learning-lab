"""Record a video of a trained model: python evaluate.py path=outputs/<env>/<algorithm>/<run-id>"""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf, DictConfig  # noqa: E402
import torch  # noqa: E402


OmegaConf.register_new_resolver(
    "random",
    lambda x: os.urandom(x).hex(),
)
OmegaConf.register_new_resolver(
    "path_safe",
    lambda x: str(x).replace(":", "_").replace("/", "_"),
)


@hydra.main(config_path="configs", config_name="eval", version_base="1.3")
def main(cfg: DictConfig):
    path = Path(__file__).parent / cfg.path
    assert path.exists(), f"Path {path} does not exist."
    assert path.is_dir(), f"Path {path} is not a directory."

    config_path = path / "config.yaml"
    assert config_path.exists(), f"Config file {config_path} does not exist."
    run_config = OmegaConf.load(config_path)

    if "parallel_envs" in run_config.env:
        del run_config.env.parallel_envs
    env = hydra.utils.call(
        run_config.env,
        enable_video=True,
        seed=cfg.seed,
    )

    torch.set_num_threads(1)

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    # ac.train.main -> ac.eval.main
    run_config.algorithm._target_ = run_config.algorithm._target_.replace("train", "eval")

    if cfg.load_step is not None:
        load_step = cfg.load_step
    else:
        # Find the latest checkpoint
        load_step = max(
            [
                int(f.stem.split("_")[-1][1:])
                for f in (path / "checkpoints").glob("model_s*.pt")
            ]
        )
    ckpt_path = path / "checkpoints" / f"model_s{load_step}.pt"
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} does not exist."

    hydra.utils.call(
        run_config.algorithm,
        env,
        ckpt_path,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
