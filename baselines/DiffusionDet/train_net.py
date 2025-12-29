# ==========================================
# Modified by Shoufa Chen
# ===========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import itertools
import weakref
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict

import torch
from fvcore.nn.precise_bn import get_bn_modules

import detectron2.utils.comm as comm
from detectron2.data.datasets import register_coco_instances
import os
from detectron2.data.datasets.builtin import register_all_coco, register_all_lvis, register_all_pascal_voc

# Register repro_10k datasets
# Assuming data structure: baselines/data/repro_10k/annotations/instances_train2017.json
try:
    # Get absolute path to baselines/data
    # This script is in baselines/DiffusionDet/
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _data_root = os.path.abspath(os.path.join(_current_dir, "../data"))
    
    register_coco_instances("repro_10k_train", {}, 
        os.path.join(_data_root, "repro_10k/annotations/instances_train2017.json"), 
        os.path.join(_data_root, "repro_10k/train2017"))
        
    register_coco_instances("repro_10k_val", {}, 
        os.path.join(_data_root, "repro_10k/annotations/instances_val2017.json"), 
        os.path.join(_data_root, "repro_10k/val2017"))
    print(f"Registered datasets from {_data_root}")
except Exception as e:
    print(f"Failed to register datasets: {e}")

# Register Detectron2 builtin datasets (COCO/LVIS/VOC) under $DETECTRON2_DATASETS (default: "datasets").
# This enables running upstream COCO/LVIS configs in this repo without additional glue code.
try:
    _d2_root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_coco(_d2_root)
    register_all_lvis(_d2_root)
    register_all_pascal_voc(_d2_root)
    print(f"Registered Detectron2 builtin datasets under {_d2_root}")
except Exception as e:
    print(f"Failed to register Detectron2 builtin datasets: {e}")

# Optional: register extra COCO-style datasets via env var (CrowdHuman / Objects365 / custom).
# Format:
#   EXTRA_COCO_DATASETS="name,json_path,image_root;name2,json2,root2"
#
# Example:
#   export EXTRA_COCO_DATASETS="crowdhuman_train,/data/crowdhuman/annotations/train.json,/data/crowdhuman/images;crowdhuman_val,/data/crowdhuman/annotations/val.json,/data/crowdhuman/images"
try:
    _extra = os.getenv("EXTRA_COCO_DATASETS", "").strip()
    if _extra:
        registered = []
        for item in _extra.split(";"):
            item = item.strip()
            if not item:
                continue
            parts = [p.strip() for p in item.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Expected 'name,json_path,image_root', got: {item!r}")
            name, json_path, image_root = parts
            register_coco_instances(name, {}, json_path, image_root)
            registered.append(name)
        print(f"Registered EXTRA_COCO_DATASETS: {registered}")
except Exception as e:
    print(f"Failed to register EXTRA_COCO_DATASETS: {e}")

from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, create_ddp_model, \
    AMPTrainer, SimpleTrainer, hooks
from detectron2.evaluation import COCOEvaluator, LVISEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import build_model

from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer


class Trainer(DefaultTrainer):
    """ Extension of the Trainer class adapted to DiffusionDet. """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()  # call grandfather's `__init__` while avoid father's `__init()`
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        ########## EMA ############
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        kwargs.update(may_get_ema_checkpointer(cfg, model))
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
            # trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        # Optional torch.compile acceleration (check.md:6.1).
        if bool(getattr(cfg.SOLVER, "TORCH_COMPILE", False)):
            if hasattr(torch, "compile"):
                mode = str(getattr(cfg.SOLVER, "TORCH_COMPILE_MODE", "default"))
                backend = str(getattr(cfg.SOLVER, "TORCH_COMPILE_BACKEND", "inductor"))
                dynamic = bool(getattr(cfg.SOLVER, "TORCH_COMPILE_DYNAMIC", False))
                try:
                    model = torch.compile(model, mode=mode, backend=backend, dynamic=dynamic)
                    logger.info("torch.compile enabled: backend=%s mode=%s dynamic=%s", backend, mode, dynamic)
                except Exception as e:
                    logger.warning("torch.compile failed; continuing without compile: %s", e)
            else:
                logger.warning("torch.compile requested but not available in this torch build; skipping.")
        # setup EMA
        may_build_model_ema(cfg, model)

        # Phase 3: distillation entries (teacher -> student).
        # Build and load a frozen teacher model and attach it to the student without registering
        # it as a submodule (to avoid checkpoint bloat).
        consistency = bool(getattr(cfg.MODEL.DiffusionDet, "CONSISTENCY_DISTILL", False))
        sampler_distill = bool(getattr(cfg.MODEL.DiffusionDet, "SAMPLER_DISTILL", False))
        teacher_w = str(getattr(cfg.MODEL.DiffusionDet, "CONSISTENCY_TEACHER_WEIGHTS", "")).strip()
        if not teacher_w and sampler_distill:
            teacher_w = str(getattr(cfg.MODEL.DiffusionDet, "SAMPLER_DISTILL_TEACHER_WEIGHTS", "")).strip()

        if consistency or sampler_distill:
            if not teacher_w:
                raise ValueError(
                    "Distillation enabled but no teacher checkpoint provided. Set one of:\n"
                    "  - MODEL.DiffusionDet.CONSISTENCY_TEACHER_WEIGHTS\n"
                    "  - MODEL.DiffusionDet.SAMPLER_DISTILL_TEACHER_WEIGHTS"
                )

            teacher_cfg = cfg.clone()
            teacher_cfg.defrost()
            teacher_cfg.MODEL.DiffusionDet.CONSISTENCY_DISTILL = False
            teacher_cfg.MODEL.DiffusionDet.CONSISTENCY_TEACHER_WEIGHTS = ""
            teacher_cfg.MODEL.DiffusionDet.SAMPLER_DISTILL = False
            teacher_cfg.MODEL.DiffusionDet.SAMPLER_DISTILL_TEACHER_WEIGHTS = ""
            if sampler_distill:
                teacher_cfg.MODEL.DiffusionDet.SAMPLE_STEP = int(
                    getattr(cfg.MODEL.DiffusionDet, "SAMPLER_DISTILL_TEACHER_SAMPLE_STEP", 20)
                )
            teacher_cfg.freeze()

            teacher = build_model(teacher_cfg)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

            DetectionCheckpointer(teacher, save_dir=cfg.OUTPUT_DIR).resume_or_load(teacher_w, resume=False)
            if sampler_distill:
                # Speed/consistency defaults for distillation: keep proposal identities stable.
                if hasattr(teacher, "use_ensemble"):
                    teacher.use_ensemble = False
                if hasattr(teacher, "box_renewal"):
                    teacher.box_renewal = False
                if hasattr(teacher, "ddim_sampling_eta"):
                    teacher.ddim_sampling_eta = float(
                        getattr(cfg.MODEL.DiffusionDet, "SAMPLER_DISTILL_TEACHER_ETA", 1.0)
                    )

            if hasattr(model, "set_teacher"):
                model.set_teacher(teacher)
            else:
                model.__dict__["teacher"] = teacher

            if consistency and sampler_distill:
                logger.info("Distillation enabled: consistency + sampler. Teacher weights: %s", teacher_w)
            elif consistency:
                logger.info("Consistency distillation enabled. Teacher weights: %s", teacher_w)
            else:
                logger.info("Sampler distillation enabled. Teacher weights: %s", teacher_w)
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if 'lvis' in dataset_name:
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        else:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            clip_group = "main"
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            if "geo_feat_" in key:
                lr = lr * float(getattr(cfg.MODEL.DiffusionDet, "GEO_FEAT_LR_MULT", 1.0))
                clip_group = "geo"
            if "geo_bias_" in key:
                lr = lr * float(getattr(cfg.MODEL.DiffusionDet, "GEO_BIAS_LR_MULT", 1.0))
                clip_group = "geo"
            if "quality_" in key:
                lr = lr * float(getattr(cfg.MODEL.DiffusionDet, "QUALITY_HEAD_LR_MULT", 1.0))
                clip_group = "quality"
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay, "clip_group": clip_group}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    main_params = []
                    geo_params = []
                    quality_params = []
                    for group in self.param_groups:
                        group_params = group.get("params", [])
                        clip_group = group.get("clip_group", "main")
                        if clip_group == "quality":
                            quality_params.extend(group_params)
                        elif clip_group == "geo":
                            geo_params.extend(group_params)
                        else:
                            main_params.extend(group_params)
                    if len(main_params) > 0:
                        torch.nn.utils.clip_grad_norm_(main_params, clip_norm_val)
                    if len(geo_params) > 0:
                        torch.nn.utils.clip_grad_norm_(geo_params, clip_norm_val)
                    if len(quality_params) > 0:
                        torch.nn.utils.clip_grad_norm_(quality_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def ema_test(cls, cfg, model, evaluators=None):
        # model with ema weights
        logger = logging.getLogger("detectron2.trainer")
        if cfg.MODEL_EMA.ENABLED:
            logger.info("Run evaluation with EMA.")
            with apply_model_ema_and_restore(model):
                results = cls.test(cfg, model, evaluators=evaluators)
        else:
            results = cls.test(cfg, model, evaluators=evaluators)
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = DiffusionDetWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        if cfg.MODEL_EMA.ENABLED:
            cls.ema_test(cfg, model, evaluators)
        else:
            res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None,  # EMA hook
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        kwargs = may_get_ema_checkpointer(cfg, model)
        if cfg.MODEL_EMA.ENABLED:
            EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                              resume=args.resume)
        else:
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                           resume=args.resume)
        res = Trainer.ema_test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
