# ---------------------------------------------------------------------------------------------------
# CLIP-DINOiser
# authors: Monika Wysoczanska, Warsaw University of Technology
# ---------------------------------------------------------------------------------------------------
from datasets import transforms

import argparse
import os
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as T
from hydra import compose, initialize
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist
from mmseg.apis import multi_gpu_test
from tqdm import tqdm

from helpers.logger import get_logger
from models import build_model
from scheduler import MultiStepLR
from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference


def get_model_dict(model):
    new_check = {}
    new_check['obj_proj.bias'] = model.state_dict()['obj_proj.bias'].cpu()
    new_check['obj_proj.weight'] = model.state_dict()['obj_proj.weight'].cpu()
    new_check['bkg_decoder.bias'] = model.state_dict()['bkg_decoder.bias'].cpu()
    new_check['bkg_decoder.weight'] = model.state_dict()['bkg_decoder.weight'].cpu()
    return new_check


def get_criterion(cfg):
    if cfg.get('loss') == 'CE':
        return torch.nn.BCEWithLogitsLoss(reduction='mean')
    else:
        raise NotImplementedError


def do_train(model, train_cfg, loaders, out_path):
    timestamp = time.time()
    date_time = datetime.fromtimestamp(timestamp)
    str_date_time = date_time.strftime("%d%m%Y-%H%M%S")

    ch_path = os.path.join(out_path, str_date_time)
    os.mkdir(ch_path)
    model.to("cuda")
    epochs = train_cfg.get("epochs", 100)
    criterion = get_criterion(train_cfg)
    optimizer = torch.optim.AdamW([{'params': model.obj_proj.parameters()},
                                   {'params': model.bkg_decoder.parameters(), 'lr': train_cfg.get('found_lr')}],
                                  lr=train_cfg.get('corr_lr'))
    scheduler = MultiStepLR(optimizer, train_cfg.get('milestones'), gamma=train_cfg.get("step_lr_gamma"), warmup=0)

    for epoch in range(epochs):
        tbar = tqdm(enumerate(loaders['train'], 0))
        for i, data in tbar:
            model.bkg_decoder.train()
            model.obj_proj.train()
            inputs = data[0].to("cuda")
            optimizer.zero_grad()
            preds_bkg, pred_corrs, clip_feats = model.forward_pass(inputs)
            pred_corrs[pred_corrs < 0] = 0.

            with torch.no_grad():
                found_pred = model.get_found_preds(inputs, resize=preds_bkg.shape[-2:])
                found_pred = (found_pred > 0.5).float()
                dino_corrs = model.get_dino_corrs(inputs).detach()

            dino_loss = criterion(pred_corrs.float().flatten(-2, -1), (dino_corrs.flatten(-2, -1) > 0).float())
            found_loss = criterion(preds_bkg.float().flatten(-2, -1), found_pred.float().flatten(-2, -1))
            loss = dino_loss + found_loss
            loss.backward()
            optimizer.step()

            tbar.set_description(f"{epoch}: {i} | {loss.item()}")
            scheduler.step()

    # save checkpoint
    model.found_model = None
    model.vit_encoder = None
    torch.save({
        'epoch': epoch,
        'model_state_dict': get_model_dict(model),
    }, os.path.join(ch_path, 'last.pt'))


@torch.no_grad()
def validate(model, cfg):
    model.eval()
    logger = get_logger()
    ret = {}
    tasks = cfg.evaluate.task

    for key in tasks:
        loader = build_seg_dataloader(build_seg_dataset(cfg.evaluate.get(key)))
        model.apply_found = False
        if key in ["voc", "coco_object"]:
            model.apply_found = True
        metric = run_val(model, loader, cfg.evaluate.get(key), logger, cfg)
        dist.broadcast_object_list(metric)
        torch.cuda.empty_cache()
        dist.barrier()
        ret[f"val/{key}_miou"] = metric[0]["mIoU"] * 100
    logger.info(ret)


def run_val(model, loader, eval_key, logger, cfg):
    model.clip_backbone.decode_head.update_vocab(loader.dataset.CLASSES)

    seg_model = build_seg_inference(
        model,
        loader.dataset,
        cfg,
        eval_key)
    seg_model.cuda()
    model.device = 'cuda'

    results = multi_gpu_test(
        model=MMDistributedDataParallel(seg_model, device_ids=[torch.cuda.current_device()]),
        data_loader=loader,
        tmpdir=None,
        gpu_collect=True,
        efficient_test=False,
        pre_eval=True,
        format_only=False,
    )

    if dist.get_rank() == 0:
        metric = [loader.dataset.evaluate(results, metric="mIoU", logger=logger)]
    else:
        metric = [None]
    return metric


def main(cfg):
    out_path = cfg.get('output')
    if out_path == "": out_path = os.getcwd()
    os.makedirs(out_path, exist_ok=True)
    dset_path = cfg.train.get('data')  # Imagenet root
    train_folder = os.path.join(dset_path, 'train')
    assert os.path.exists(train_folder), 'Empty dataset path'
    logger = get_logger(cfg)
    logger.info(f"Running CLIP-DINOiser training")

    # set up the ImageFolder loader
    im_size = cfg.train.get('im_size', 448)
    num_workers = cfg.train.get('num_workers', 4)
    transforms = [T.ToTensor(), T.Resize(im_size), T.RandomCrop(im_size), T.RandomHorizontalFlip(p=0.5),
                  T.ColorJitter(0.5)]
    train_dataset = torchvision.datasets.ImageFolder(train_folder, transform=T.Compose(transforms))

    if cfg.train.get("ds_size", None) is not None:
        # if subset: SAMPLE
        indices = np.random.choice(list(range(len(train_dataset))), cfg.train.ds_size)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train.batch_size,
                                                   num_workers=num_workers,
                                                   sampler=torch.utils.data.SubsetRandomSampler(indices))
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train.batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True)

    classes = ['']  # dummy text query - we do not use text queries for the training
    model = build_model(cfg.model, class_names=classes)
    model.load_teachers()
    mp.set_start_method("fork", force=True)
    init_dist("pytorch", )
    rank, world_size = get_dist_info()
    logger.info(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")

    cudnn.benchmark = True
    do_train(model, cfg.train, {"train": train_loader}, out_path=out_path)
    logger.info(f"Training finished. Running evaluation...")

    ## we can remove dino and found
    model.found_model = None
    model.vit_encoder = None

    # run full evaluation
    validate(model, cfg)


def parse_args():
    parser = argparse.ArgumentParser(
        description='CLIP-DINOiser training procedure')
    parser.add_argument('config', help='config file path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    initialize(config_path="configs", version_base=None)
    cfg = compose(config_name=args.config)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    main(cfg)
