from mmcv.utils import Registry

MODELS = Registry('models')
from omegaconf import OmegaConf


def build_model(config, class_names):
    model = MODELS.build(OmegaConf.to_container(config, resolve=True),
                         default_args={'class_names': class_names})
    return model