# ------------------------------------------------------------------------------
# Modified from TCL
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------

import logging
import os.path as osp

from mmcv.utils import get_logger as get_root_logger
from termcolor import colored

logger_name = None


def get_logger(cfg=None, log_level=logging.INFO):
    global logger_name
    if cfg is None:
        return get_root_logger(logger_name)

    # creating logger
    name = cfg.model_name
    output = cfg.output
    logger_name = name

    logger = get_root_logger(name, osp.join(output, "log.txt"), log_level=log_level, file_mode="a")
    logger.propagate = False

    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    color_fmt = (
        colored("[%(asctime)s %(name)s]", "green")
        + colored("(%(filename)s %(lineno)d)", "yellow")
        + ": %(levelname)s %(message)s"
    )

    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S"))

        if isinstance(handler, logging.FileHandler):
            handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    return logger
