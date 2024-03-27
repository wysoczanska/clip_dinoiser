import mmcv
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
from mmseg.datasets import build_dataloader, build_dataset

MODELS = Registry('models', parent=MMCV_MODELS)

SEGMENTORS = MODELS
from .clip_dinoiser_eval import DinoCLIP_Infrencer


def build_seg_dataset(config):
    """Build a dataset from config."""
    cfg = mmcv.Config.fromfile(config)
    dataset = build_dataset(cfg.data.test)
    return dataset


def build_seg_dataloader(dataset, dist=True):
    # batch size is set to 1 to handle varying image size (due to different aspect ratio)
    if dist:
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=dist,
            shuffle=False,
            persistent_workers=True,
            pin_memory=False,
        )
    else:
        data_loader = build_dataloader(
            dataset=dataset,
            samples_per_gpu=1,
            workers_per_gpu=1,
            dist=dist,
            shuffle=False,
            persistent_workers=True,
            pin_memory=False,
        )

    return data_loader


def build_seg_inference(
        model,
        dataset,
        config,
        seg_config,
):
    dset_cfg = mmcv.Config.fromfile(seg_config)  # dataset config
    classnames = dataset.CLASSES
    kwargs = dict()
    if hasattr(dset_cfg, "test_cfg"):
        kwargs["test_cfg"] = dset_cfg.test_cfg

    seg_model = DinoCLIP_Infrencer(model, num_classes=len(classnames), **kwargs, **config.evaluate)
    seg_model.CLASSES = dataset.CLASSES
    seg_model.PALETTE = dataset.PALETTE

    return seg_model
