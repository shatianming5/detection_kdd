# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_diffusiondet_config(cfg):
    """
    Add config for DiffusionDet
    """
    cfg.MODEL.DiffusionDet = CN()
    cfg.MODEL.DiffusionDet.NUM_CLASSES = 80
    cfg.MODEL.DiffusionDet.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.DiffusionDet.NHEADS = 8
    cfg.MODEL.DiffusionDet.DROPOUT = 0.0
    cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DiffusionDet.ACTIVATION = 'relu'
    cfg.MODEL.DiffusionDet.HIDDEN_DIM = 256
    cfg.MODEL.DiffusionDet.NUM_CLS = 1
    cfg.MODEL.DiffusionDet.NUM_REG = 3
    cfg.MODEL.DiffusionDet.NUM_HEADS = 6
    # Self-attention implementation in the proposal graph.
    # - "torch": torch.nn.MultiheadAttention (baseline; supports returning attn weights for topo loss)
    # - "sdpa":  torch.nn.functional.scaled_dot_product_attention (uses FlashAttention-2 kernel when available)
    cfg.MODEL.DiffusionDet.SELF_ATTN_IMPL = "torch"
    # When SELF_ATTN_IMPL=="sdpa": optionally force a backend ("auto" | "flash" | "mem_efficient" | "math").
    cfg.MODEL.DiffusionDet.SELF_ATTN_SDPA_BACKEND = "auto"
    # Ablations: disable proposal self-attention (no node interaction).
    cfg.MODEL.DiffusionDet.DISABLE_SELF_ATTN = False
    # Geometry-aware attention bias (C2O-GND MVP).
    # When enabled, adds a non-learned geometry bias to proposal self-attention.
    cfg.MODEL.DiffusionDet.GEO_BIAS = False
    # distance: negative RBF on normalized center distance.
    # iou: positive bias on box IoU (diagonal set to 0).
    cfg.MODEL.DiffusionDet.GEO_BIAS_TYPE = "distance"
    # constant: always apply bias; linear: scale by (1 - t/T)^p (stronger at low-noise steps).
    cfg.MODEL.DiffusionDet.GEO_BIAS_SCHEDULE = "constant"
    cfg.MODEL.DiffusionDet.GEO_BIAS_TIME_POWER = 1.0
    # Used when GEO_BIAS_SCHEDULE == "threshold": enable bias only when diffusion t <= threshold.
    cfg.MODEL.DiffusionDet.GEO_BIAS_T_THRESHOLD = 0
    # If enabled, learn a per-head scale in [-GEO_BIAS_SCALE, GEO_BIAS_SCALE] (tanh-parameterized).
    cfg.MODEL.DiffusionDet.GEO_BIAS_LEARNABLE_SCALE = False
    cfg.MODEL.DiffusionDet.GEO_BIAS_SCALE = 1.0
    # Distance normalization uses box sqrt(area); sigma controls decay speed.
    cfg.MODEL.DiffusionDet.GEO_BIAS_SIGMA = 2.0
    # Optional clamp for the distance normalization denominator (sqrt(area) in pixels).
    # 0.0 disables clamping (keeps original behavior).
    cfg.MODEL.DiffusionDet.GEO_BIAS_MIN_NORM = 0.0
    # Optional learnable geometry bias network g_phi (check.md:4.3 / 6.2).
    # - When GEO_BIAS_TYPE == "mlp", compute a pairwise bias from relative geometry features.
    cfg.MODEL.DiffusionDet.GEO_BIAS_NORM = 1000.0
    cfg.MODEL.DiffusionDet.GEO_BIAS_USE_LOG_WH = True
    cfg.MODEL.DiffusionDet.GEO_BIAS_INPUT_CLIP = 10.0
    cfg.MODEL.DiffusionDet.GEO_BIAS_REL_INCLUDE_IOU = False
    cfg.MODEL.DiffusionDet.GEO_BIAS_MLP_HIDDEN_DIM = 64
    cfg.MODEL.DiffusionDet.GEO_BIAS_OUT_TANH = True
    # Optional sparsification (k-NN style): keep top-k neighbors per node; 0 disables.
    cfg.MODEL.DiffusionDet.GEO_BIAS_TOPK = 0
    # Reduce learning rate for geo bias parameters (same spirit as GEO_FEAT_LR_MULT).
    cfg.MODEL.DiffusionDet.GEO_BIAS_LR_MULT = 1.0

    # Geometry feature injection (safer than hard attention-logit bias).
    # When enabled, injects a learnable geometry-conditioned signal into proposal features.
    # The projection is zero-initialized after weight init so it starts as a no-op.
    cfg.MODEL.DiffusionDet.GEO_FEAT = False
    cfg.MODEL.DiffusionDet.GEO_FEAT_ENCODER = "mlp"  # "mlp" | "linear"
    cfg.MODEL.DiffusionDet.GEO_FEAT_SCALE = 0.1
    # Reduce learning rate for geo feature injection parameters to improve stability.
    # This multiplier is applied on top of SOLVER.BASE_LR (and backbone multiplier if applicable).
    cfg.MODEL.DiffusionDet.GEO_FEAT_LR_MULT = 1.0
    # Extra learnable gate in (0, 1) to let the model decide whether to use GEO_FEAT.
    # This scales the GEO_FEAT output as: GEO_FEAT_SCALE * sigmoid(alpha_param).
    cfg.MODEL.DiffusionDet.GEO_FEAT_ALPHA_INIT = 0.1
    cfg.MODEL.DiffusionDet.GEO_FEAT_NORM = 1000.0
    cfg.MODEL.DiffusionDet.GEO_FEAT_USE_LOG_WH = True
    cfg.MODEL.DiffusionDet.GEO_FEAT_INCLUDE_T = True
    cfg.MODEL.DiffusionDet.GEO_FEAT_TARGET = "proposal"
    # When GEO_FEAT_TARGET == "proposal":
    # - "add": add residual into proposal features (affects q/k/v in self-attn).
    # - "qk_pos": use geometry residual as positional encoding for q/k only (value unchanged).
    cfg.MODEL.DiffusionDet.GEO_FEAT_PROPOSAL_MODE = "qk_pos"
    # Only enable geometry injection after a given training iteration (0 disables).
    # Useful to let the baseline denoising dynamics settle before introducing new signals.
    cfg.MODEL.DiffusionDet.GEO_FEAT_TRAIN_START_ITER = 0
    cfg.MODEL.DiffusionDet.GEO_FEAT_TRAIN_WARMUP_ITERS = 0
    cfg.MODEL.DiffusionDet.GEO_FEAT_MLP_HIDDEN_DIM = 128
    # Optional: geometry-conditioned message passing with relative encoding (O(N^2) like attention).
    cfg.MODEL.DiffusionDet.GEO_FEAT_REL = False
    cfg.MODEL.DiffusionDet.GEO_FEAT_REL_MLP_HIDDEN_DIM = 64
    cfg.MODEL.DiffusionDet.GEO_FEAT_REL_INCLUDE_IOU = False
    cfg.MODEL.DiffusionDet.GEO_FEAT_REL_TANH = True
    # Stabilizers:
    # - Input clamp for geometry scalars (0 disables clamping).
    # - Optional tanh on the MLP output to keep injected deltas bounded.
    cfg.MODEL.DiffusionDet.GEO_FEAT_INPUT_CLIP = 10.0
    cfg.MODEL.DiffusionDet.GEO_FEAT_OUT_TANH = True
    # Relative message normalization: "softmax" | "mean" | "none"
    cfg.MODEL.DiffusionDet.GEO_FEAT_REL_NORM = "softmax"
    cfg.MODEL.DiffusionDet.GEO_FEAT_SCHEDULE = "linear"
    cfg.MODEL.DiffusionDet.GEO_FEAT_TIME_POWER = 2.0
    cfg.MODEL.DiffusionDet.GEO_FEAT_T_THRESHOLD = 0

    # Phase 2(A): Label-state (masked corruption / absorbing unk) - minimal skeleton.
    # This provides a discrete label "state" input c_t in {0..K-1, unk} for each proposal.
    # Training: for proposals originating from GT boxes, keep the true label with prob keep_prob(t),
    # otherwise replace with unk (absorbing state). Inference: start from all unk.
    # Default is disabled to keep baseline behavior unchanged.
    cfg.MODEL.DiffusionDet.LABEL_STATE = False
    cfg.MODEL.DiffusionDet.LABEL_STATE_SCALE = 0.1
    cfg.MODEL.DiffusionDet.LABEL_STATE_ALPHA_INIT = 0.1
    cfg.MODEL.DiffusionDet.LABEL_STATE_PROJ_ZERO_INIT = True
    # If True, inject label embedding relative to the unk state so that "all unk" is an exact no-op.
    # This reduces train-time interference when keep_prob(t) is small (label_state mostly unk).
    cfg.MODEL.DiffusionDet.LABEL_STATE_RELATIVE_TO_UNK = False
    # keep_prob schedule: "sqrt_alphas_cumprod" | "alphas_cumprod" | "linear" | "constant"
    cfg.MODEL.DiffusionDet.LABEL_STATE_KEEP_PROB_SCHEDULE = "sqrt_alphas_cumprod"
    cfg.MODEL.DiffusionDet.LABEL_STATE_KEEP_PROB_CONST = 0.0
    cfg.MODEL.DiffusionDet.LABEL_STATE_KEEP_PROB_POWER = 1.0
    cfg.MODEL.DiffusionDet.LABEL_STATE_KEEP_PROB_MIN = 0.0
    cfg.MODEL.DiffusionDet.LABEL_STATE_FORCE_UNK_INFER = True

    # Phase 2(B): D3PM-style discrete forward process (strong form).
    # When enabled, builds an explicit forward transition (Q̄_t) over {0..K-1, unk}
    # and generates c_t from c_0 with selectable kernel.
    cfg.MODEL.DiffusionDet.LABEL_D3PM = False
    # "mask": absorbing unk; "uniform": mixes with uniform over all states.
    cfg.MODEL.DiffusionDet.LABEL_D3PM_KERNEL = "mask"
    # If True, pass a probability distribution (N, num_proposals, K+1) into the head.
    # If False, sample hard indices from the distribution.
    cfg.MODEL.DiffusionDet.LABEL_D3PM_USE_DISTRIBUTION = True
    # Inference: update label-state during diffusion sampling (recommended for LABEL_D3PM).
    # When disabled, inference keeps label_state fixed (typically all unk), which can create
    # a train/infer mismatch when using soft distributions.
    cfg.MODEL.DiffusionDet.LABEL_D3PM_INFER_UPDATE = True

    # Phase 3: Quality/Energy head + energy-guided sampling (optional).
    # The quality head predicts a [0,1] "box quality" score (e.g. IoU with matched GT) for each proposal.
    # Default is disabled to keep baseline behavior unchanged.
    cfg.MODEL.DiffusionDet.QUALITY_HEAD = False
    cfg.MODEL.DiffusionDet.QUALITY_HEAD_HIDDEN_DIM = 256
    cfg.MODEL.DiffusionDet.QUALITY_HEAD_INCLUDE_T = True
    cfg.MODEL.DiffusionDet.QUALITY_HEAD_USE_LOG_WH = True
    cfg.MODEL.DiffusionDet.QUALITY_BOX_NORM = 1000.0
    # Loss for quality head: "bce" (with soft IoU targets) | "l1" | "mse"
    cfg.MODEL.DiffusionDet.QUALITY_LOSS_TYPE = "bce"
    cfg.MODEL.DiffusionDet.QUALITY_LOSS_WEIGHT = 1.0
    cfg.MODEL.DiffusionDet.QUALITY_HEAD_LR_MULT = 1.0

    # Energy-guided sampling based on predicted quality (inference only).
    # When scale == 0, sampling is identical to baseline.
    cfg.MODEL.DiffusionDet.QUALITY_GUIDANCE_SCALE = 0.0
    cfg.MODEL.DiffusionDet.QUALITY_GUIDANCE_TOPK = 50
    cfg.MODEL.DiffusionDet.QUALITY_GUIDANCE_SCORE_WEIGHT = True
    # Gradient normalization: "none" | "proposal" | "global"
    cfg.MODEL.DiffusionDet.QUALITY_GUIDANCE_GRAD_NORM = "proposal"
    # Guidance timing and Langevin-style updates:
    # - mode="final": apply only at the final denoise step (current default behavior)
    # - mode="all": apply at every diffusion step
    # - mode="threshold": apply when t <= QUALITY_GUIDANCE_T_THRESHOLD
    cfg.MODEL.DiffusionDet.QUALITY_GUIDANCE_MODE = "final"
    cfg.MODEL.DiffusionDet.QUALITY_GUIDANCE_T_THRESHOLD = 0
    # Inner Langevin steps per diffusion step (>=1).
    cfg.MODEL.DiffusionDet.QUALITY_GUIDANCE_LANGEVIN_STEPS = 1
    # Optional extra Gaussian noise added after each Langevin step (0 disables).
    cfg.MODEL.DiffusionDet.QUALITY_GUIDANCE_LANGEVIN_NOISE = 0.0
    # Step size schedule over diffusion time: "constant" | "linear"
    cfg.MODEL.DiffusionDet.QUALITY_GUIDANCE_STEP_SCHEDULE = "constant"
    cfg.MODEL.DiffusionDet.QUALITY_GUIDANCE_TIME_POWER = 1.0

    # Inference-time score reweighting using predicted quality.
    # When enabled, scales classification scores by sigmoid(quality)^QUALITY_SCORE_POWER.
    # This is deterministic and does not require differentiating through ROIAlign/box coords.
    cfg.MODEL.DiffusionDet.QUALITY_SCORE_REWEIGHT = False
    cfg.MODEL.DiffusionDet.QUALITY_SCORE_POWER = 1.0

    # Phase 3: consistency distillation (teacher -> student).
    # This is a lightweight training entry that adds an extra regression loss between
    # student and a frozen teacher on the same noisy proposals/time step.
    cfg.MODEL.DiffusionDet.CONSISTENCY_DISTILL = False
    cfg.MODEL.DiffusionDet.CONSISTENCY_TEACHER_WEIGHTS = ""
    cfg.MODEL.DiffusionDet.CONSISTENCY_DISTILL_BOX_WEIGHT = 0.0
    cfg.MODEL.DiffusionDet.CONSISTENCY_DISTILL_CLS_WEIGHT = 0.0
    cfg.MODEL.DiffusionDet.CONSISTENCY_DISTILL_CLS_TEMPERATURE = 1.0

    # Phase 3: sampler distillation (teacher multi-step -> student fewer-step).
    # Target use-case: distill a strong teacher (e.g. SAMPLE_STEP=20) into a fast student (SAMPLE_STEP=1).
    cfg.MODEL.DiffusionDet.SAMPLER_DISTILL = False
    cfg.MODEL.DiffusionDet.SAMPLER_DISTILL_TEACHER_WEIGHTS = ""
    cfg.MODEL.DiffusionDet.SAMPLER_DISTILL_TEACHER_SAMPLE_STEP = 20
    cfg.MODEL.DiffusionDet.SAMPLER_DISTILL_STUDENT_SAMPLE_STEP = 1
    # Teacher sampling stochasticity (DDIM eta). 0.0 makes teacher targets deterministic given init noise.
    cfg.MODEL.DiffusionDet.SAMPLER_DISTILL_TEACHER_ETA = 1.0
    # Optional student sampling stochasticity when distilling to a multi-step student.
    # Only used when SAMPLER_DISTILL_STUDENT_SAMPLE_STEP > 1 and the distillation loss runs the student's sampler.
    cfg.MODEL.DiffusionDet.SAMPLER_DISTILL_STUDENT_ETA = 0.0
    # Optional: distill only the top-k proposals (by teacher max-class score) to avoid over-regularizing background.
    # 0 means use all proposals.
    cfg.MODEL.DiffusionDet.SAMPLER_DISTILL_TOPK = 0
    cfg.MODEL.DiffusionDet.SAMPLER_DISTILL_BOX_WEIGHT = 0.0
    cfg.MODEL.DiffusionDet.SAMPLER_DISTILL_CLS_WEIGHT = 0.0
    cfg.MODEL.DiffusionDet.SAMPLER_DISTILL_CLS_TEMPERATURE = 1.0

    # Phase 1: optional torch.compile acceleration (check.md:6.1).
    # This is experimental and may not work for all models/ops.
    cfg.SOLVER.TORCH_COMPILE = False
    cfg.SOLVER.TORCH_COMPILE_MODE = "default"  # "default" | "reduce-overhead" | "max-autotune"
    cfg.SOLVER.TORCH_COMPILE_BACKEND = "inductor"  # e.g. "inductor" | "aot_eager"
    cfg.SOLVER.TORCH_COMPILE_DYNAMIC = False

    # Dynamic Conv.
    cfg.MODEL.DiffusionDet.NUM_DYNAMIC = 2
    cfg.MODEL.DiffusionDet.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.DiffusionDet.CLASS_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.GIOU_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.L1_WEIGHT = 5.0
    cfg.MODEL.DiffusionDet.DEEP_SUPERVISION = True
    cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT = 0.1
    # check.md:5.1/5.2 — Optional quality-aware losses (default off; keeps baseline behavior).
    # Classification loss type:
    # - "focal": existing sigmoid focal (one-hot targets)
    # - "qfl": Quality Focal Loss with soft targets y∈[0,1] (e.g. IoU as target for matched class)
    cfg.MODEL.DiffusionDet.CLS_LOSS_TYPE = "focal"
    cfg.MODEL.DiffusionDet.QFL_BETA = 2.0
    # IoU-aware regression weighting (Varifocal-style idea): weight matched box losses by IoU^p.
    # 0 disables. Recommend p in {1,2} for experiments.
    cfg.MODEL.DiffusionDet.BOX_LOSS_IOU_WEIGHT_POWER = 0.0
    # Phase 2: optional topology loss on proposal graph.
    # Computes a supervision signal between predicted proposal self-attention (graph adjacency)
    # and GT adjacency (e.g., IoU>threshold) over the matched GT instances.
    cfg.MODEL.DiffusionDet.GRAPH_TOPO_LOSS_WEIGHT = 0.0
    cfg.MODEL.DiffusionDet.GRAPH_TOPO_IOU_THRESH = 0.1
    # "row": row-normalize adjacency like attention; "none": raw 0/1 target.
    cfg.MODEL.DiffusionDet.GRAPH_TOPO_TARGET_NORM = "row"

    # Focal Loss.
    cfg.MODEL.DiffusionDet.USE_FOCAL = True
    cfg.MODEL.DiffusionDet.USE_FED_LOSS = False
    cfg.MODEL.DiffusionDet.ALPHA = 0.25
    cfg.MODEL.DiffusionDet.GAMMA = 2.0
    cfg.MODEL.DiffusionDet.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.DiffusionDet.OTA_K = 5

    # Diffusion
    cfg.MODEL.DiffusionDet.SNR_SCALE = 2.0
    cfg.MODEL.DiffusionDet.SAMPLE_STEP = 1
    cfg.MODEL.DiffusionDet.DIFFUSION_TIMESTEPS = 1000
    # Phase 2: anisotropic Gaussian noise in box diffusion space.
    # When enabled, forward q_sample uses noise ~ N(0, diag(sigma^2)) with per-dim std:
    #   [sigma_xy, sigma_xy, sigma_w, sigma_h]
    # This also scales the random noise used during DDIM sampling so the forward/reverse
    # distributions remain consistent.
    cfg.MODEL.DiffusionDet.ANISO_NOISE = False
    cfg.MODEL.DiffusionDet.ANISO_NOISE_SIGMA_XY = 1.0
    cfg.MODEL.DiffusionDet.ANISO_NOISE_SIGMA_W = 1.0
    cfg.MODEL.DiffusionDet.ANISO_NOISE_SIGMA_H = 1.0

    # Inference
    cfg.MODEL.DiffusionDet.USE_NMS = True

    # Swin Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])
