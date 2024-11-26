import os
import yaml
from carlutils.carlconfig import get_cfg


def load_config(yaml_file):
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if yaml_file is not None and os.path.exists(yaml_file):
        with open(yaml_file, 'r') as config_file:
            config_dict = yaml.safe_load(config_file)
        cfg.update(config_dict)

    cfg.EVAL.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
    cfg.EVAL.NUM_FRAMES = cfg.TRAIN.NUM_FRAMES
    return cfg