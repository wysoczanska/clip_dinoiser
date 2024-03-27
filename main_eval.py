# ---------------------------------------------------------------------------------------------------
# CLIP-DINOiser
# authors: Monika Wysoczanska, Warsaw University of Technology
# ---------------------------------------------------------------------------------------------------
# modified from TCL (https://github.com/kakaobrain/tcl/) Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ---------------------------------------------------------------------------------------------------

import os
import argparse
import datasets.transforms

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra import compose, initialize
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmseg.apis import multi_gpu_test

from helpers.logger import get_logger
from models import build_model
from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference


@torch.no_grad()
def evaluate(cfg, val_loaders):
    logger = get_logger()
    ret = {}

    for key, loader in val_loaders.items():

        logger.info(f"### Validation dataset: {key}")
        CLASSES = loader.dataset.CLASSES
        logger.info(f"Creating model:{cfg.model.type}")
        model = build_model(cfg.model, class_names=CLASSES)
        model.apply_found = False
        if key in ["voc", "coco_object"]:
            model.apply_found = True

        check_path = 'checkpoints/last.pt'
        check = torch.load(check_path)['model_state_dict']
        model.load_state_dict(check, strict=False)
        model.cuda()
        model.device = "cuda"
        model.eval()

        miou, metrics = validate_seg(cfg, cfg.evaluate.get(key), loader, model)
        logger.info(f"[{key}] mIoU of {len(loader.dataset)} test images: {miou:.2f}%")
        ret[f"val/{key}_miou"] = miou

    ret["val/avg_miou"] = np.mean([v for k, v in ret.items() if "miou" in k])
    return ret


@torch.no_grad()
def validate_seg(config, seg_config, data_loader, model):
    logger = get_logger()
    dist.barrier()
    model.eval()
    seg_model = build_seg_inference(
        model,
        data_loader.dataset,
        config,
        seg_config,
    )

    mmddp_model = MMDistributedDataParallel(
        seg_model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False
    )
    mmddp_model.eval()

    results = multi_gpu_test(
        model=mmddp_model,
        data_loader=data_loader,
        tmpdir=None,
        gpu_collect=True,
        efficient_test=False,
        pre_eval=True,
        format_only=False,
    )

    if dist.get_rank() == 0:
        metric = [data_loader.dataset.evaluate(results, metric="mIoU", logger=logger)]
    else:
        metric = [None]

    dist.broadcast_object_list(metric)
    miou_result = metric[0]["mIoU"] * 100
    torch.cuda.empty_cache()
    dist.barrier()
    return miou_result, metric


def main(cfg):
    mp.set_start_method("fork", force=True)
    init_dist("pytorch")
    rank, world_size = get_dist_info()
    print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")

    dist.barrier()
    set_random_seed(cfg.seed, use_rank_shift=True)
    cudnn.benchmark = True

    logger = get_logger(cfg)

    val_loaders = {}
    for key in cfg.evaluate.task:
        loader = build_seg_dataloader(build_seg_dataset(cfg.evaluate.get(key)))
        val_loaders[key] = loader
    res = evaluate(cfg, val_loaders)
    logger.info(res)
    dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='config file path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    initialize(config_path="configs", version_base=None)
    cfg = compose(config_name=args.config)
    main(cfg)
