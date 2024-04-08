import torch

import nms_1d_cpu_vg


class NMSop(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, segs, scores,
        iou_thresh, min_score, max_num_segs
    ):
        filter_by_score = min_score > 0
        if filter_by_score:
            mask = scores > min_score
            segs, scores = segs[mask], scores[mask]

        # indices sorted in descending order
        idx = nms_1d_cpu_vg.nms(
            segs.contiguous().cpu(),
            scores.contiguous().cpu(),
            iou_thresh=float(iou_thresh)
        )

        if max_num_segs > 0:
            idx = idx[:min(max_num_segs, len(idx))]

        sorted_segs = segs[idx].contiguous()
        sorted_scores = scores[idx].contiguous()

        return sorted_segs, sorted_scores


class SoftNMSop(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, segs, scores,
        iou_thresh, sigma, min_score, method, max_num_segs
    ):
        out = segs.new_empty((len(segs), 3), device='cpu')

        # indices sorted in descending order
        idx = nms_1d_cpu_vg.softnms(
            segs.contiguous().cpu(),
            scores.contiguous().cpu(),
            out,
            iou_thresh=float(iou_thresh),
            sigma=float(sigma),
            min_score=float(min_score),
            method=int(method),
        )

        num_segs = len(idx)
        if max_num_segs > 0:
            num_segs = min(num_segs, max_num_segs)

        sorted_segs = out[:num_segs, :2].contiguous()
        sorted_scores = out[:num_segs, 2].contiguous()

        return sorted_segs, sorted_scores


def segment_voting(
    nms_segs,
    all_segs,
    all_scores,
    iou_thresh,
):
    """
    Refine localization results by combining highly overlaping segments.

    Args:
        nms_segs (n1, 2): segments filtered by NMS.
        all_segs (n2, 2): pre-filtered segments.
        all_scores (n2,): pre-filtered scores.
        iou_thresh (float): IOU overlap threshold.

    Returns:
        refined_segs (n1, 2): refined segments.
    """
    nms_segs = nms_segs[:, None]    # (n1, 1, 2)
    all_segs = all_segs[None, :]    # (1, n2, 2)

    # intersection: (n1, n2)
    left = torch.maximum(nms_segs[..., 0], all_segs[..., 0])
    right = torch.minimum(nms_segs[..., 1], all_segs[..., 1])
    overlap = (right - left).clamp(min=0)

    # union: (n1, n2)
    nms_seg_lens = nms_segs[..., 1] - nms_segs[..., 0]
    all_seg_lens = all_segs[..., 1] - all_segs[..., 0]
    union = nms_seg_lens + all_seg_lens - overlap

    # IOU
    iou = overlap / union

    # collect highly overlapping segments and combine them
    weights = (iou >= iou_thresh).float() * all_scores[None]
    weights /= torch.sum(weights, dim=1, keepdim=True)
    refined_segs = weights @ all_segs[0]

    return refined_segs


def batched_nms(
    segs,
    scores,
    iou_thresh,
    min_score,
    max_num_segs,
    mode='soft_nms',
    sigma=0.5,
    voting_thresh=0.75,
):
    if len(segs) == 0:
        return torch.zeros(0, 2), torch.zeros(0)

    if mode is not None:
        if mode == 'nms':
            nms_segs, nms_scores = NMSop.apply(
                segs, scores, iou_thresh,
                min_score, max_num_segs
            )
        elif mode == 'soft_nms':
            nms_segs, nms_scores = SoftNMSop.apply(
                segs, scores, iou_thresh,
                sigma, min_score, 2, max_num_segs
            )
        else:
            raise NotImplementedError('invalid NMS mode')

        if voting_thresh > 0:
            nms_segs = segment_voting(
                nms_segs,
                segs,
                scores,
                voting_thresh,
            )
    else:
        nms_segs, nms_scores = segs, scores

    idx = nms_scores.argsort(descending=True)
    max_num_segs = min(max_num_segs, len(nms_segs))
    nms_segs = nms_segs[idx[:max_num_segs]]
    nms_scores = nms_scores[idx[:max_num_segs]]

    return nms_segs, nms_scores
