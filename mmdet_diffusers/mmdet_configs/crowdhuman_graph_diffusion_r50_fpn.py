_base_ = [
    "mmdet::_base_/default_runtime.py",
]

import inspect
import os
import sys

__cfg_file = inspect.currentframe().f_code.co_filename  # type: ignore[union-attr]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__cfg_file), "..", "..")))

custom_imports = dict(imports=["mmdet_diffusers.mmdet3"], allow_failed_imports=False)

dataset_type = "CrowdHumanDataset"
data_root = os.environ.get("CROWDHUMAN_ROOT", "data/CrowdHuman/").rstrip("/") + "/"
backend_args = None

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1400, 800), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "flip", "flip_direction"),
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(1400, 800), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

num_classes = 1
num_nodes = 500
unk_id = num_classes

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    collate_fn=dict(type="coco_graph_collate", num_nodes=num_nodes, unk_id=unk_id, shuffle_nodes=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotation_train.odgt",
        data_prefix=dict(img="Images/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    collate_fn=dict(type="coco_graph_collate", num_nodes=num_nodes, unk_id=unk_id, shuffle_nodes=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotation_val.odgt",
        data_prefix=dict(img="Images/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CrowdHumanMetric",
    ann_file=data_root + "annotation_val.odgt",
    metric=["AP", "MR", "JI"],
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
    sampling_timesteps=1,
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
    use_geo_bias=True,
    geo_bias_type="mlp",
    geo_bias_scale=1.0,
    use_label_state=True,
    label_state_scale=0.1,
    use_quality_head=True,
    quality_loss_weight=1.0,
    graph_topo_loss_weight=0.0,
    score_thr=0.05,
    nms_iou_thr=0.6,
    max_per_img=300,
)

max_epochs = 30
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=800),
    dict(type="MultiStepLR", begin=0, end=max_epochs, by_epoch=True, milestones=[24, 27], gamma=0.1),
]

optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05))

auto_scale_lr = dict(enable=False, base_batch_size=16)
