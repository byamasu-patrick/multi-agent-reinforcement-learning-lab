import os
from pathlib import Path
import sys

# make the `ac` package importable however this script is launched; this also runs
# in the sub-processes that AsyncVectorEnv spawns on Windows
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf, DictConfig  # noqa: E402
import torch  # noqa: E402

OmegaConf.register_new_resolver(
    "random",
    lambda x: os.urandom(x).hex(),
)
# env names such as "lbforaging:Foraging-8x8-2p-3f-v3" contain characters that are not
# allowed in Windows directory names
OmegaConf.register_new_resolver(
    "path_safe",
    lambda x: str(x).replace(":", "_").replace("/", "_"),
)


@hydra.main(config_path="configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    logger = hydra.utils.instantiate(cfg.logger, cfg=cfg, _recursive_=False)

    env = hydra.utils.call(cfg.env, seed=cfg.seed)

    # Use singular env for evaluation/ recording
    if "parallel_envs" in cfg.env:
        del cfg.env.parallel_envs
    eval_env = hydra.utils.call(
        cfg.env,
        enable_video=True if cfg.algorithm.video_interval else False,
        seed=cfg.seed,
    )

    torch.set_num_threads(1)

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
    else:
        logger.warning("No seed has been set.")

    assert cfg.env.time_limit is not None, "Time limit must be set."
    hydra.utils.call(
        cfg.algorithm,
        env,
        eval_env,
        logger,
        time_limit=cfg.env.time_limit,
        _recursive_=False,
    )

    return logger.get_state()


if __name__ == "__main__":
    main()
