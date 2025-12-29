_base_ = [
    "mmdet::_base_/default_runtime.py",
]

import inspect
import os
import sys

__cfg_file = inspect.currentframe().f_code.co_filename  # type: ignore[union-attr]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__cfg_file), "..", "..")))

custom_imports = dict(imports=["mmdet_diffusers.mmdet3"], allow_failed_imports=False)

dataset_type = "CocoDataset"
data_root = os.environ.get("COCO_ROOT", "data/coco/")
backend_args = None

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
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

num_classes = 80
num_nodes = 500
unk_id = num_classes

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    collate_fn=dict(type="coco_graph_collate", num_nodes=num_nodes, unk_id=unk_id, shuffle_nodes=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/instances_train2017.json",
        data_prefix=dict(img="train2017/"),
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
        ann_file="annotations/instances_val2017.json",
        data_prefix=dict(img="val2017/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/instances_val2017.json",
    metric="bbox",
    format_only=False,
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
        type="SwinTransformer",
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type="Pretrained",
            checkpoint=os.environ.get(
                "SWIN_BASE_CKPT",
                "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth",
            ),
        ),
    ),
    neck=dict(type="FPN", in_channels=[128, 256, 512, 1024], out_channels=256, num_outs=5),
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
    use_geo_bias=False,
    use_label_state=False,
    use_quality_head=False,
    graph_topo_loss_weight=0.0,
    score_thr=0.05,
    nms_iou_thr=0.6,
    max_per_img=100,
)

max_epochs = 30
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[24, 27],
        gamma=0.1,
    ),
]

optim_wrapper = dict(
    type="OptimWrapper",
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
    optimizer=dict(type="AdamW", lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
)

auto_scale_lr = dict(enable=False, base_batch_size=16)

