_base_ = [
    "mmdet::_base_/default_runtime.py",
]

import inspect
import os
import sys

__cfg_file = inspect.currentframe().f_code.co_filename  # type: ignore[union-attr]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__cfg_file), "..", "..")))

custom_imports = dict(imports=["mmdet_diffusers.mmdet3"], allow_failed_imports=False)

dataset_type = "LVISV1Dataset"
data_root = os.environ.get("LVIS_ROOT", "/data/tiasha/datasets/lvis_v1/").rstrip("/") + "/"
backend_args = None

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="RandomChoiceResize",
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768), (1333, 800)],
        keep_ratio=True,
    ),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

num_classes = 1203
num_nodes = 500
unk_id = num_classes

train_dataloader = dict(
    batch_size=2,
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    collate_fn=dict(type="coco_graph_collate", num_nodes=num_nodes, unk_id=unk_id, shuffle_nodes=True),
    dataset=dict(
        type="ClassBalancedDataset",
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file="annotations/lvis_v1_train.json",
            data_prefix=dict(img=""),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args,
        ),
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    collate_fn=dict(type="coco_graph_collate", num_nodes=num_nodes, unk_id=unk_id, shuffle_nodes=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/lvis_v1_val.json",
        data_prefix=dict(img=""),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="LVISMetric",
    ann_file=data_root + "annotations/lvis_v1_val.json",
    metric=["bbox"],
    backend_args=backend_args,
)
test_evaluator = val_evaluator

model = dict(
    type="GraphDiffusionDetector",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(type="FPN", in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5),
    num_classes=num_classes,
    num_proposals=num_nodes,
    diffusion_timesteps=1000,
    sampling_timesteps=10,
    ddim_sampling_eta=1.0,
    box_scale=2.0,
    hidden_dim=256,
    dim_feedforward=2048,
    nhead=8,
    dropout=0.0,
    activation="relu",
    num_heads=6,
    deep_supervision=True,
    pooler_resolution=7,
    roi_featmap_strides=(4, 8, 16, 32),
    dim_dynamic=64,
    num_dynamic=2,
    num_cls_layers=1,
    num_reg_layers=1,
    use_geo_bias=False,
    use_label_state=False,
    use_quality_head=False,
    quality_loss_weight=0.0,
    quality_guidance_scale=0.0,
    graph_topo_loss_weight=0.0,
    cls_loss_type="focal",
    focal_alpha=0.25,
    focal_gamma=2.0,
    prior_prob=0.01,
    init_head_xavier=True,
    loss_cls_weight=2.0,
    loss_bbox_weight=5.0,
    loss_giou_weight=2.0,
    no_object_weight=0.1,
    use_dynamic_k_matching=True,
    ota_k=5,
    score_thr=0.0,
    nms_iou_thr=0.6,
    max_per_img=300,
)

max_iters = 150000
train_cfg = dict(type="IterBasedTrainLoop", max_iters=max_iters, val_interval=max_iters + 1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=2000),
    dict(type="MultiStepLR", by_epoch=False, begin=0, end=max_iters, milestones=[120000, 140000], gamma=0.1),
]

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=5e-5, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.5, norm_type=2),
)

default_hooks = dict(
    logger=dict(interval=50),
    checkpoint=dict(by_epoch=False, interval=10000, max_keep_ckpts=3),
)

custom_hooks = [
    dict(type="CheckInvalidLossHook", interval=50),
]

env_cfg = dict(cudnn_benchmark=True)

load_from = os.environ.get("LVIS_LOAD_FROM", "").strip() or None

_disable_val = os.environ.get("MMDET_DISABLE_VAL", "").strip().lower() in {"1", "true", "yes"}
if _disable_val:
    val_dataloader = None
    val_evaluator = None
    val_cfg = None
    test_dataloader = None
    test_evaluator = None
    test_cfg = None
