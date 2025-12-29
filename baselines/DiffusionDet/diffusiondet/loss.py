# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit
import torchvision.ops as ops
from .util import box_ops
from .util.misc import get_world_size, is_dist_avail_and_initialized
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou


class SetCriterionDynamicK(nn.Module):
    """ This class computes the loss for DiffusionDet.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, cfg, num_classes, matcher, weight_dict, eos_coef, losses, use_focal):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal = use_focal
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        if self.use_fed_loss:
            self.fed_loss_num_classes = 50
            from detectron2.data.detection_utils import get_fed_loss_cls_weights
            cls_weight_fun = lambda: get_fed_loss_cls_weights(dataset_names=cfg.DATASETS.TRAIN, freq_weight_power=cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER)  # noqa
            fed_loss_cls_weights = cls_weight_fun()
            assert (
                    len(fed_loss_cls_weights) == self.num_classes
            ), "Please check the provided fed_loss_cls_weights. Their size should match num_classes"
            self.register_buffer("fed_loss_cls_weights", fed_loss_cls_weights)

        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.DiffusionDet.ALPHA
            self.focal_loss_gamma = cfg.MODEL.DiffusionDet.GAMMA
        else:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)
        self.quality_loss_type = str(getattr(cfg.MODEL.DiffusionDet, "QUALITY_LOSS_TYPE", "bce")).lower()
        self.cls_loss_type = str(getattr(cfg.MODEL.DiffusionDet, "CLS_LOSS_TYPE", "focal")).lower()
        self.qfl_beta = float(getattr(cfg.MODEL.DiffusionDet, "QFL_BETA", 2.0))
        self.box_loss_iou_weight_power = float(getattr(cfg.MODEL.DiffusionDet, "BOX_LOSS_IOU_WEIGHT_POWER", 0.0))
        if self.cls_loss_type not in {"focal", "qfl"}:
            raise ValueError(f"Unsupported CLS_LOSS_TYPE={self.cls_loss_type!r}")
        if self.box_loss_iou_weight_power < 0.0:
            raise ValueError("BOX_LOSS_IOU_WEIGHT_POWER must be >= 0")
        self.graph_topo_iou_thresh = float(getattr(cfg.MODEL.DiffusionDet, "GRAPH_TOPO_IOU_THRESH", 0.1))
        self.graph_topo_target_norm = str(getattr(cfg.MODEL.DiffusionDet, "GRAPH_TOPO_TARGET_NORM", "row")).lower()

    # copy-paste from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py#L356
    def get_fed_loss_classes(self, gt_classes, num_fed_loss_classes, num_classes, weight):
        """
        Args:
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
            num_fed_loss_classes: minimum number of classes to keep when calculating federated loss.
            Will sample negative classes if number of unique gt_classes is smaller than this value.
            num_classes: number of foreground classes
            weight: probabilities used to sample negative classes
        Returns:
            Tensor:
                classes to keep when calculating the federated loss, including both unique gt
                classes and sampled negative classes.
        """
        unique_gt_classes = torch.unique(gt_classes)
        prob = unique_gt_classes.new_ones(num_classes + 1).float()
        prob[-1] = 0
        if len(unique_gt_classes) < num_fed_loss_classes:
            prob[:num_classes] = weight.float().clone()
            prob[unique_gt_classes] = 0
            sampled_negative_classes = torch.multinomial(
                prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
            )
            fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])
        else:
            fed_loss_classes = unique_gt_classes
        return fed_loss_classes

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        batch_size = len(targets)

        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        src_logits_list = []
        target_classes_o_list = []
        # target_classes[idx] = target_classes_o
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_src_logits = src_logits[batch_idx]
            target_classes_o = targets[batch_idx]["labels"]
            target_classes[batch_idx, valid_query] = target_classes_o[gt_multi_idx]

            src_logits_list.append(bz_src_logits[valid_query])
            target_classes_o_list.append(target_classes_o[gt_multi_idx])

        if not (self.use_focal or self.use_fed_loss):
            raise NotImplementedError("Only sigmoid-style classification is supported in this repo.")

        num_boxes = torch.cat(target_classes_o_list).shape[0] if len(target_classes_o_list) != 0 else 1

        if self.cls_loss_type == "focal":
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], self.num_classes + 1],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            gt_classes = torch.argmax(target_classes_onehot, dim=-1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]

            src_logits_flat = src_logits.flatten(0, 1)
            target_flat = target_classes_onehot.flatten(0, 1)
            if self.use_focal:
                cls_loss = sigmoid_focal_loss_jit(
                    src_logits_flat,
                    target_flat,
                    alpha=self.focal_loss_alpha,
                    gamma=self.focal_loss_gamma,
                    reduction="none",
                )
            else:
                cls_loss = F.binary_cross_entropy_with_logits(src_logits_flat, target_flat, reduction="none")
        else:
            # check.md:5.2 — Quality Focal Loss (QFL) with soft targets.
            # Target for matched (query, gt) pairs: y = IoU(pred_box, gt_box) on the matched class.
            assert "pred_boxes" in outputs, "QFL requires outputs['pred_boxes'] for IoU targets."
            src_boxes = outputs["pred_boxes"]  # (bs, num_queries, 4) absolute (x1,y1,x2,y2)

            target_scores = torch.zeros(
                (src_logits.shape[0], src_logits.shape[1], self.num_classes),
                device=src_logits.device,
                dtype=src_logits.dtype,
            )
            for batch_idx in range(batch_size):
                valid_query = indices[batch_idx][0]
                gt_multi_idx = indices[batch_idx][1]
                if gt_multi_idx.numel() == 0:
                    continue
                q_idx = valid_query.nonzero(as_tuple=False).squeeze(1)
                if q_idx.numel() == 0:
                    continue
                cls = targets[batch_idx]["labels"][gt_multi_idx].to(device=src_logits.device, dtype=torch.long)

                bz_src_boxes = src_boxes[batch_idx][valid_query]
                bz_tgt_boxes_xyxy = targets[batch_idx]["boxes_xyxy"][gt_multi_idx].to(device=src_boxes.device)
                iou = ops.box_iou(bz_src_boxes, bz_tgt_boxes_xyxy).diag().clamp(0.0, 1.0).detach()
                target_scores[batch_idx, q_idx, cls] = iou.to(dtype=target_scores.dtype)

            # QFL: |y - p|^beta * BCEWithLogits(logit, y)
            pred_sigmoid = torch.sigmoid(src_logits)
            beta = max(float(self.qfl_beta), 0.0)
            weight = torch.abs(target_scores - pred_sigmoid).pow(beta)
            cls_loss = F.binary_cross_entropy_with_logits(src_logits, target_scores, reduction="none") * weight

            gt_classes = target_classes  # (bs, num_queries) with background id == num_classes
            src_logits_flat = src_logits.flatten(0, 1)
            cls_loss = cls_loss.flatten(0, 1)

        if self.use_fed_loss:
            K = self.num_classes
            N = src_logits_flat.shape[0]
            fed_loss_classes = self.get_fed_loss_classes(
                gt_classes,
                num_fed_loss_classes=self.fed_loss_num_classes,
                num_classes=K,
                weight=self.fed_loss_cls_weights,
            )
            fed_loss_classes_mask = fed_loss_classes.new_zeros(K + 1)
            fed_loss_classes_mask[fed_loss_classes] = 1
            fed_loss_classes_mask = fed_loss_classes_mask[:K]
            weight_mask = fed_loss_classes_mask.view(1, K).expand(N, K).float()
            loss_ce = torch.sum(cls_loss * weight_mask) / num_boxes
        else:
            loss_ce = torch.sum(cls_loss) / num_boxes

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        # idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes']

        batch_size = len(targets)
        pred_box_list = []
        pred_norm_box_list = []
        tgt_box_list = []
        tgt_box_xyxy_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_image_whwh = targets[batch_idx]['image_size_xyxy']
            bz_src_boxes = src_boxes[batch_idx]
            bz_target_boxes = targets[batch_idx]["boxes"]  # normalized (cx, cy, w, h)
            bz_target_boxes_xyxy = targets[batch_idx]["boxes_xyxy"]  # absolute (x1, y1, x2, y2)
            pred_box_list.append(bz_src_boxes[valid_query])
            pred_norm_box_list.append(bz_src_boxes[valid_query] / bz_image_whwh)  # normalize (x1, y1, x2, y2)
            tgt_box_list.append(bz_target_boxes[gt_multi_idx])
            tgt_box_xyxy_list.append(bz_target_boxes_xyxy[gt_multi_idx])

        if len(pred_box_list) != 0:
            src_boxes = torch.cat(pred_box_list)
            src_boxes_norm = torch.cat(pred_norm_box_list)  # normalized (x1, y1, x2, y2)
            target_boxes = torch.cat(tgt_box_list)
            target_boxes_abs_xyxy = torch.cat(tgt_box_xyxy_list)
            num_boxes = src_boxes.shape[0]

            losses = {}
            # require normalized (x1, y1, x2, y2)
            loss_bbox = F.l1_loss(src_boxes_norm, box_cxcywh_to_xyxy(target_boxes), reduction="none")

            # loss_giou = giou_loss(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes, target_boxes_abs_xyxy))
            if self.box_loss_iou_weight_power > 0.0:
                # check.md:5.1 — IoU-aware regression weighting (Varifocal-style idea).
                iou = ops.box_iou(src_boxes, target_boxes_abs_xyxy).diag().clamp(0.0, 1.0).detach()
                w = (iou + 1e-3).pow(float(self.box_loss_iou_weight_power)).to(dtype=loss_giou.dtype)
                loss_bbox = loss_bbox * w[:, None]
                loss_giou = loss_giou * w

            losses["loss_bbox"] = loss_bbox.sum() / num_boxes
            losses["loss_giou"] = loss_giou.sum() / num_boxes
        else:
            losses = {'loss_bbox': outputs['pred_boxes'].sum() * 0,
                      'loss_giou': outputs['pred_boxes'].sum() * 0}

        return losses

    def loss_quality(self, outputs, targets, indices, num_boxes):
        """Auxiliary quality loss.

        Predict a continuous "quality" score per proposal and supervise it with IoU of matched pairs.
        The supervision is applied only on matched proposals (selected_query==True).
        """
        assert 'pred_quality' in outputs
        assert 'pred_boxes' in outputs
        src_quality = outputs['pred_quality']  # (bs, num_queries)
        src_boxes = outputs['pred_boxes']  # (bs, num_queries, 4) absolute (x1, y1, x2, y2)

        batch_size = len(targets)
        pred_list = []
        target_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_src_boxes = src_boxes[batch_idx][valid_query]
            bz_tgt_boxes_xyxy = targets[batch_idx]["boxes_xyxy"][gt_multi_idx]
            # Treat IoU as a supervision target (stop-grad). Otherwise this loss would
            # backprop through IoU into the box regression path and destabilize training.
            iou = ops.box_iou(bz_src_boxes, bz_tgt_boxes_xyxy).diag().clamp(0.0, 1.0).detach()
            pred_list.append(src_quality[batch_idx][valid_query])
            target_list.append(iou.to(dtype=src_quality.dtype))

        if len(pred_list) == 0:
            return {'loss_quality': outputs['pred_quality'].sum() * 0}

        pred = torch.cat(pred_list)
        target = torch.cat(target_list)
        num_pos = max(int(target.numel()), 1)

        if self.quality_loss_type == "bce":
            loss = F.binary_cross_entropy_with_logits(pred, target, reduction="sum") / num_pos
        elif self.quality_loss_type == "l1":
            loss = F.l1_loss(torch.sigmoid(pred), target, reduction="sum") / num_pos
        elif self.quality_loss_type == "mse":
            loss = F.mse_loss(torch.sigmoid(pred), target, reduction="sum") / num_pos
        else:
            raise ValueError(f"Unsupported QUALITY_LOSS_TYPE={self.quality_loss_type!r}")

        return {'loss_quality': loss}

    def loss_graph(self, outputs, targets, indices, num_boxes, matched_ids=None):
        """Topology loss on the proposal graph.

        Supervise the predicted proposal self-attention adjacency (from the last RCNN head)
        to match GT adjacency (IoU>threshold) over the matched GT instances.

        Args:
            outputs["pred_attn"]: (bs, num_queries, num_queries) attention weights.
            matched_ids: list[Tensor], each is (num_gt,) giving the best-matched query id per GT.
        """
        if "pred_attn" not in outputs or matched_ids is None:
            return {"loss_graph": outputs["pred_boxes"].sum() * 0}

        attn = outputs["pred_attn"]  # (bs, num_queries, num_queries)
        bs = attn.shape[0]
        loss = attn.sum() * 0
        pairs = 0

        for b in range(bs):
            gt_boxes = targets[b].get("boxes_xyxy", None)
            if gt_boxes is None:
                continue
            num_gt = int(gt_boxes.shape[0])
            if num_gt <= 1:
                continue

            qids = matched_ids[b]
            if qids.numel() != num_gt:
                # Fallback: use selected queries (may include duplicates). Keep this robust.
                sel = indices[b][0].nonzero(as_tuple=False).squeeze(1)
                if sel.numel() < 2:
                    continue
                qids = sel[:num_gt]

            qids = qids.to(device=attn.device, dtype=torch.long)
            a_pred = attn[b].to(dtype=torch.float32)[qids][:, qids]  # (num_gt, num_gt)

            iou = ops.box_iou(gt_boxes.to(device=attn.device), gt_boxes.to(device=attn.device)).to(dtype=torch.float32)
            a_tgt = (iou > float(self.graph_topo_iou_thresh)).to(dtype=torch.float32)

            eye = torch.eye(num_gt, device=attn.device, dtype=torch.float32)
            a_pred = a_pred * (1.0 - eye)
            a_tgt = a_tgt * (1.0 - eye)

            if self.graph_topo_target_norm == "row":
                denom_tgt = a_tgt.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                a_tgt = a_tgt / denom_tgt
                denom_pred = a_pred.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                a_pred = a_pred / denom_pred
            elif self.graph_topo_target_norm != "none":
                raise ValueError(f"Unsupported GRAPH_TOPO_TARGET_NORM={self.graph_topo_target_norm!r}")

            loss = loss + F.mse_loss(a_pred, a_tgt, reduction="sum")
            pairs += num_gt * (num_gt - 1)

        if pairs <= 0:
            return {"loss_graph": outputs["pred_boxes"].sum() * 0}
        return {"loss_graph": loss / float(pairs)}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'quality': self.loss_quality,
            'graph': self.loss_graph,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, matched_ids = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == "graph":
                losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, matched_ids=matched_ids))
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, _ = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks' or loss == "graph":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class HungarianMatcherDynamicK(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-k (dynamic) matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_mask: float = 1, use_focal: bool = False):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.ota_k = cfg.MODEL.DiffusionDet.OTA_K
        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.DiffusionDet.ALPHA
            self.focal_loss_gamma = cfg.MODEL.DiffusionDet.GAMMA
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0,  "all costs cant be 0"

    def forward(self, outputs, targets):
        """ simOTA for detr"""
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            # We flatten to compute the cost matrices in a batch
            if self.use_focal or self.use_fed_loss:
                out_prob = outputs["pred_logits"].sigmoid()  # [batch_size, num_queries, num_classes]
                out_bbox = outputs["pred_boxes"]  # [batch_size,  num_queries, 4]
            else:
                out_prob = outputs["pred_logits"].softmax(-1)  # [batch_size, num_queries, num_classes]
                out_bbox = outputs["pred_boxes"]  # [batch_size, num_queries, 4]

            indices = []
            matched_ids = []
            assert bs == len(targets)
            for batch_idx in range(bs):
                bz_boxes = out_bbox[batch_idx]  # [num_proposals, 4]
                bz_out_prob = out_prob[batch_idx]
                bz_tgt_ids = targets[batch_idx]["labels"]
                num_insts = len(bz_tgt_ids)
                if num_insts == 0:  # empty object in key frame
                    non_valid = torch.zeros(bz_out_prob.shape[0]).to(bz_out_prob) > 0
                    indices_batchi = (non_valid, torch.arange(0, 0).to(bz_out_prob))
                    matched_qidx = torch.arange(0, 0).to(bz_out_prob)
                    indices.append(indices_batchi)
                    matched_ids.append(matched_qidx)
                    continue

                bz_gtboxs = targets[batch_idx]['boxes']  # [num_gt, 4] normalized (cx, xy, w, h)
                bz_gtboxs_abs_xyxy = targets[batch_idx]['boxes_xyxy']
                fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
                    box_xyxy_to_cxcywh(bz_boxes),  # absolute (cx, cy, w, h)
                    box_xyxy_to_cxcywh(bz_gtboxs_abs_xyxy),  # absolute (cx, cy, w, h)
                    expanded_strides=32
                )

                pair_wise_ious = ops.box_iou(bz_boxes, bz_gtboxs_abs_xyxy)

                # Compute the classification cost.
                if self.use_focal:
                    alpha = self.focal_loss_alpha
                    gamma = self.focal_loss_gamma
                    neg_cost_class = (1 - alpha) * (bz_out_prob ** gamma) * (-(1 - bz_out_prob + 1e-8).log())
                    pos_cost_class = alpha * ((1 - bz_out_prob) ** gamma) * (-(bz_out_prob + 1e-8).log())
                    cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
                elif self.use_fed_loss:
                    # focal loss degenerates to naive one
                    neg_cost_class = (-(1 - bz_out_prob + 1e-8).log())
                    pos_cost_class = (-(bz_out_prob + 1e-8).log())
                    cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
                else:
                    cost_class = -bz_out_prob[:, bz_tgt_ids]

                # Compute the L1 cost between boxes
                # image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])
                # image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
                # image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])

                bz_image_size_out = targets[batch_idx]['image_size_xyxy']
                bz_image_size_tgt = targets[batch_idx]['image_size_xyxy_tgt']

                bz_out_bbox_ = bz_boxes / bz_image_size_out  # normalize (x1, y1, x2, y2)
                bz_tgt_bbox_ = bz_gtboxs_abs_xyxy / bz_image_size_tgt  # normalize (x1, y1, x2, y2)
                cost_bbox = torch.cdist(bz_out_bbox_, bz_tgt_bbox_, p=1)

                cost_giou = -generalized_box_iou(bz_boxes, bz_gtboxs_abs_xyxy)

                # Final cost matrix
                cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + 100.0 * (~is_in_boxes_and_center)
                # cost = (cost_class + 3.0 * cost_giou + 100.0 * (~is_in_boxes_and_center))  # [num_query,num_gt]
                cost[~fg_mask] = cost[~fg_mask] + 10000.0

                # if bz_gtboxs.shape[0]>0:
                indices_batchi, matched_qidx = self.dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs.shape[0])

                indices.append(indices_batchi)
                matched_ids.append(matched_qidx)

        return indices, matched_ids

    def get_in_boxes_info(self, boxes, target_gts, expanded_strides):
        xy_target_gts = box_cxcywh_to_xyxy(target_gts)  # (x1, y1, x2, y2)

        anchor_center_x = boxes[:, 0].unsqueeze(1)
        anchor_center_y = boxes[:, 1].unsqueeze(1)

        # whether the center of each anchor is inside a gt box
        b_l = anchor_center_x > xy_target_gts[:, 0].unsqueeze(0)
        b_r = anchor_center_x < xy_target_gts[:, 2].unsqueeze(0)
        b_t = anchor_center_y > xy_target_gts[:, 1].unsqueeze(0)
        b_b = anchor_center_y < xy_target_gts[:, 3].unsqueeze(0)
        # (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4 [300,num_gt] ,
        is_in_boxes = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_boxes_all = is_in_boxes.sum(1) > 0  # [num_query]
        # in fixed center
        center_radius = 2.5
        # Modified to self-adapted sampling --- the center size depends on the size of the gt boxes
        # https://github.com/dulucas/UVO_Challenge/blob/main/Track1/detection/mmdet/core/bbox/assigners/rpn_sim_ota_assigner.py#L212
        b_l = anchor_center_x > (target_gts[:, 0] - (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_r = anchor_center_x < (target_gts[:, 0] + (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_t = anchor_center_y > (target_gts[:, 1] - (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)
        b_b = anchor_center_y < (target_gts[:, 1] + (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)

        is_in_centers = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_centers_all = is_in_centers.sum(1) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (is_in_boxes & is_in_centers)

        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt):
        matching_matrix = torch.zeros_like(cost)  # [300,num_gt]
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = self.ota_k

        # Take the sum of the predicted value and the top 10 iou of gt with the largest iou as dynamic_k
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(1)

        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
            matching_matrix[anchor_matching_gt > 1] *= 0
            matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1

        while (matching_matrix.sum(0) == 0).any():
            num_zero_gt = (matching_matrix.sum(0) == 0).sum()
            matched_query_id = matching_matrix.sum(1) > 0
            cost[matched_query_id] += 100000.0
            unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:, gt_idx])
                matching_matrix[:, gt_idx][pos_idx] = 1.0
            if (matching_matrix.sum(1) > 1).sum() > 0:  # If a query matches more than one gt
                _, cost_argmin = torch.min(cost[anchor_matching_gt > 1],
                                           dim=1)  # find gt for these queries with minimal cost
                matching_matrix[anchor_matching_gt > 1] *= 0  # reset mapping relationship
                matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1  # keep gt with minimal cost

        assert not (matching_matrix.sum(0) == 0).any()
        selected_query = matching_matrix.sum(1) > 0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        assert selected_query.sum() == len(gt_indices)

        cost[matching_matrix == 0] = cost[matching_matrix == 0] + float('inf')
        matched_query_id = torch.min(cost, dim=0)[1]

        return (selected_query, gt_indices), matched_query_id
