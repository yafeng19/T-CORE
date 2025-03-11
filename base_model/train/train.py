
import os
import math
import torch
import logging
import argparse
import base_model.distributed as distributed

from functools import partial
from fvcore.common.checkpoint import PeriodicCheckpointer

from base_model.utils.config import setup
from base_model.logging import MetricLogger
from base_model.fsdp import FSDPCheckpointer
from base_model.train.arch import SSLArch
from base_model.utils.utils import CosineScheduler
from base_model.data import SamplerType, make_data_loader, make_dataset_for_videos
from base_model.data import DataAugmentationVideo, MaskingGenerator, collate_data_and_cast_with_aux_use_past_future_frames

torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger("TCoRe")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("TCoRe training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--no-resume", action="store_true", help="Whether to not attempt to resume from the checkpoint directory. ")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument("opts", help="Modify config options at the end of the command. ".strip(), default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--output-dir", "--output_dir", default="", type=str, help="Output directory to save logs and checkpoints")
    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def build_aux_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    aux_lr = dict(
            base_value=cfg.optim["lr"]*cfg.auxiliary.lr_mul,
            final_value=cfg.optim["min_lr"],
            total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
            warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
            start_warmup_value=0,
    )
    aux_lr_schedule = CosineScheduler(**aux_lr)
    return (
        aux_lr_schedule
    )
        
def build_mask_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    min_mask_ratio = dict(
        base_value=cfg.ibot["min_mask_ratio_range"][0],
        final_value=cfg.ibot["min_mask_ratio_range"][1],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    max_mask_ratio = dict(
        base_value=cfg.ibot["max_mask_ratio_range"][0],
        final_value=cfg.ibot["max_mask_ratio_range"][1],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    mask_sample_prob = dict(
        base_value=cfg.ibot["mask_sample_prob_range"][0],
        final_value=cfg.ibot["mask_sample_prob_range"][1],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )

    min_mask_ratio_schedule = CosineScheduler(**min_mask_ratio)
    max_mask_ratio_schedule = CosineScheduler(**max_mask_ratio)
    mask_sample_prob_schedule = CosineScheduler(**mask_sample_prob)

    return (
        min_mask_ratio_schedule,
        max_mask_ratio_schedule,
        mask_sample_prob_schedule
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier

            
def apply_optim_scheduler_with_aux(optimizer, lr, wd, last_layer_lr, aux_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier

        is_aux_layer = param_group["is_aux_layer"]
        param_group["lr"] = aux_lr if is_aux_layer else lr


def do_test(cfg, model, iteration):
    new_state_dict = model.teacher.state_dict()
    iterstring = str(iteration)
    eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
    teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
 
    if distributed.is_main_process():
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)

    res_dict = {}
    if distributed.is_main_process():
        # Implementation for specific downstream task evaluation
        pass

    # synchronize between processes
    if distributed.is_enabled():
        torch.distributed.barrier()

    return res_dict



def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    if cfg.auxiliary.use_auxiliary:
        (
            aux_lr_schedule
        ) = build_aux_schedulers(cfg)

    if cfg.ibot.mask_change:
        (
            min_mask_ratio_schedule,
            max_mask_ratio_schedule,
            mask_sample_prob_schedule
        ) = build_mask_schedulers(cfg)

    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2

    data_transform_video = DataAugmentationVideo(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches= img_size // patch_size * img_size // patch_size,
    )

    collate_fn_select = collate_data_and_cast_with_aux_use_past_future_frames

    collate_fn = partial(
        collate_fn_select,
        mask_ratio_tuple=(cfg.ibot.min_mask_ratio_range[0], cfg.ibot.max_mask_ratio_range[0]) if cfg.ibot.mask_change else cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_prob_range[0] if cfg.ibot.mask_change else cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    if not cfg.ibot.mask_change:
        min_mask_ratio = cfg.ibot.mask_ratio_min_max[0]
        max_mask_ratio = cfg.ibot.mask_ratio_min_max[1]
        mask_sample_prob = cfg.ibot.mask_sample_probability

    dataset = make_dataset_for_videos(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform_video,
        target_transform=(lambda _: ()),
        past_offset_range = cfg.train.past_offset_range,
        current_range = cfg.train.current_range,
        future_offset_range = cfg.train.future_offset_range
    )

    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,
        sampler_type=sampler_type,
        sampler_advance=0,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    tb_file = os.path.join(cfg.train.output_dir, "logs")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file, tb_file=tb_file)
    header = "Training"

    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        # update min/max mask ratio and mask sample probability
        if cfg.ibot.mask_change:
            min_mask_ratio = min_mask_ratio_schedule[iteration]
            max_mask_ratio = max_mask_ratio_schedule[iteration]
            mask_sample_prob = mask_sample_prob_schedule[iteration]

            # redefine collate_fn with updated values
            collate_fn_ = partial(
                collate_fn_select,
                mask_ratio_tuple=(min_mask_ratio, max_mask_ratio),
                mask_probability=mask_sample_prob,
                n_tokens=n_tokens,
                mask_generator=mask_generator,
                dtype=inputs_dtype,
            )
            data_loader.collate_fn = collate_fn_

            
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        if cfg.auxiliary.use_auxiliary:
            aux_lr = aux_lr_schedule[iteration]
            apply_optim_scheduler_with_aux(optimizer, lr, wd, last_layer_lr, aux_lr)
        else:
            apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp, iteration=iteration)

        # clip gradients
        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
                if cfg.auxiliary.use_auxiliary:
                    for v in model.auxiliary.values():
                        v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
                if cfg.auxiliary.use_auxiliary:
                    for v in model.auxiliary.values():
                        v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform teacher EMA update
        model.update_teacher(mom)

        # logging
        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}
        
        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update_iteration(iteration)
        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        if cfg.auxiliary.use_auxiliary:
            metric_logger.update(aux_lr=aux_lr)
            metric_logger.update(min_mask_ratio=min_mask_ratio)
            metric_logger.update(max_mask_ratio=max_mask_ratio)
            metric_logger.update(mask_sample_prob=mask_sample_prob)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        if iteration % cfg.train.tb_log_freq == 0:
            if cfg.auxiliary.use_auxiliary:
                metric_logger.update_tensorboard(
                    ['lr', 'wd', 'mom', 'last_layer_lr', 'aux_lr', 'min_mask_ratio', 'max_mask_ratio', 'mask_sample_prob', 'current_batch_size'],
                    [lr, wd, mom, last_layer_lr, aux_lr, min_mask_ratio, max_mask_ratio, mask_sample_prob, current_batch_size],
                    'setting'
                )
            else:
                metric_logger.update_tensorboard(
                    ['lr', 'wd', 'mom', 'last_layer_lr', 'current_batch_size'],
                    [lr, wd, mom, last_layer_lr, current_batch_size],
                    'setting'
                )
            metric_logger.update_tensorboard(
                ['total_loss'] + list(loss_dict_reduced.keys()),
                [losses_reduced] + list(loss_dict_reduced.values()),
                'train'
            )

        # checkpointing and testing
        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            eval_metrics = do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
            metric_logger.update_tensorboard(list(eval_metrics.keys()), list(eval_metrics.values()), 'test')
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)

    model = SSLArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
