# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import functools
import logging
import os
import sys
from typing import Optional

import base_model.distributed as distributed
from .helpers import MetricLogger, SmoothedValue



@functools.lru_cache()
def _configure_logger(
    name: Optional[str] = None,
    *,
    level: int = logging.DEBUG,
    output: Optional[str] = None,
):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    fmt_prefix = "%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] "
    fmt_message = "%(message)s"
    fmt = fmt_prefix + fmt_message
    datefmt = "%Y%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if distributed.is_main_process():
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.handlers.clear()
        logger.addHandler(handler)

    if output:
        if os.path.splitext(output)[-1] in (".txt", ".log"):
            filename = output
        else:
            filename = os.path.join(output, "logs", "log.txt")

        if not distributed.is_main_process():
            global_rank = distributed.get_global_rank()
            filename = filename + ".rank{}".format(global_rank)

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        handler = logging.StreamHandler(open(filename, "a"))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def setup_logging(
    output: Optional[str] = None,
    *,
    name: Optional[str] = None,
    level: int = logging.DEBUG,
    capture_warnings: bool = True,
) -> None:
    
    logging.captureWarnings(capture_warnings)
    _configure_logger(name, level=level, output=output)
