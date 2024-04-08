from collections import OrderedDict
from copy import deepcopy
from functools import partial
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from .data_utils import trivial_batch_collator, worker_init_reset_seed
from .tokenizer import make_tokenizer


datasets = dict()
def register_dataset(name):
    def decorator(module):
        datasets[name] = module
        return module
    return decorator


class BaseDataset(Dataset):

    def __init__(
        self,
        split,                  # data split, a tuple/list allowing concat of subsets
        is_training,            # whether in training mode
        
        anno_file,              # annotation json file
        vid_feat_dir,           # video feature directory
        text_feat_dir,          # text feature directory
        ext_score_dir,          # external score directory
        tokenizer,              # tokenizer (optional)

        max_vid_len,            # max video length (#clips) in training
        max_text_len,           # max text length (#tokens) in training
        clip_size,              # number of frames per clip / feature
        clip_stride,            # temporal stride of clips (in frame)
        downsample_rate=1,      # down-sampling rate for video features
        to_fixed_len=False,     # whether to resize video features to max length
        
        normalize_vid=False,    # whether to normalize video features to unit length
        normalize_text=False,   # whether to normalize text features to unit length
        normalize_scores=True,  # whether to normalize external score using sigmoid
        temperature=1.0,        # sigmoid temperature for score normalization

        crop_ratio=(0.9, 1.0),  # random cropping of video features in training
        trunc_thresh=0.5,       # threshold for event truncation in training
        max_num_text=None,      # max number of text queries per video in training
        
        group_method="greedy",  # text grouping method ("greedy" | "random" | "all")
        num_epochs=1,           # number of epochs
    ):
        super(BaseDataset, self).__init__()

        assert os.path.exists(anno_file)
        if not isinstance(split, (list, tuple)):
            split = (split, )
        if not isinstance(vid_feat_dir, (list, tuple)):
            vid_feat_dir = (vid_feat_dir, )
        assert all([os.path.isdir(d) for d in vid_feat_dir])
        if tokenizer is None:
            assert text_feat_dir is not None, (
                "text features must be given if tokenizer is not specified"
            )
        assert isinstance(downsample_rate, int) and downsample_rate >= 1
        if crop_ratio is not None:
            assert isinstance(crop_ratio, (list, tuple))

        self.split = split
        self.is_training = is_training
        self.epoch = 0  # this must be updated upon starting a new epoch

        self.anno_file = anno_file
        self.vid_feat_dir = vid_feat_dir
        self.text_feat_dir = text_feat_dir
        self.ext_score_dir = ext_score_dir
        self.tokenizer = tokenizer

        self.max_vid_len = max_vid_len
        self.max_text_len = max_text_len
        self.clip_size = clip_size
        self.clip_stride = clip_stride * downsample_rate
        self.downsample_rate = downsample_rate
        self.to_fixed_len = to_fixed_len

        self.normalize_vid = normalize_vid
        self.normalize_text = normalize_text
        self.normalize_scores = normalize_scores
        self.temperature = temperature

        self.crop_ratio = crop_ratio
        self.trunc_thresh = trunc_thresh
        self.max_num_text = max_num_text

        self.vid_dict, self.text_dict = self._parse_annotations()
        
        self.group_method = group_method
        self.num_epochs = num_epochs

    def _parse_annotations(self):
        with open(self.anno_file, 'r') as f:
            anno = json.load(f)

        # combine data from all splits
        anno_db = dict()
        for s in self.split:
            assert s in anno, 'split [{:s}] does not exist'.format(s)
            anno_db.update(anno[s])

        vid_dict, text_dict = OrderedDict(), OrderedDict()
        for key, value in anno_db.items():
            if 'annotations' not in value:
                continue

            fps, num_frames = float(value['fps']), int(value['num_frames'])
            if 'duration' in value:
                duration = float(value['duration'])
            else:
                duration = num_frames / fps
            
            if 'num_clips' in value:
                num_clips = (
                    value['num_clips'] + self.downsample_rate - 1
                ) // self.downsample_rate
            else:
                num_clips = None

            text_ids, segments = tuple(), tuple()
            for s, pair in enumerate(value['annotations']):
                start = max(float(pair['segment'][0]), 0)
                end = min(float(pair['segment'][1]), duration)
                seg_len = end - start
                if seg_len <= 0:
                    continue
                segment = (start, end)

                text = pair['sentence'].strip()
                text_id = pair.get('sentence_id', key + '_{:04d}'.format(s))
                text_ids += (text_id, )
                segments += (segment, )

                text_dict[text_id] = {
                    'text'      : text,
                    'segment'   : np.array(segment)[None],
                    'text_idx'  : s,
                    'vid_id'    : key,
                }
            
            if len(text_ids) == 0:
                continue

            vid_dict[key] = {
                'fps'       : fps,
                'num_frames': num_frames,
                'num_clips' : num_clips,
                'duration'  : duration,
                'text_ids'  : text_ids,
                'segments'  : np.array(segments),
            }

        return vid_dict, text_dict

    def _load_vid_feats(self, vid_id):
        try:
            vid_feat_files = [os.path.join(d, vid_id + '.npy') \
                for d in self.vid_feat_dir]
            vid_feats = [np.load(f).astype(np.float32) for f in vid_feat_files]
        except:
            raise ValueError(
                'failed to load features for video {:s}'.format(vid_id)
            )

        # assume features from different sources are apporoximately aligned
        # (flow features may be one unit shorter than RGB features)
        if len(vid_feats) > 1:
            feat_lens = [len(x) for x in vid_feats]
            max_len, min_len = max(feat_lens), min(feat_lens)
            assert max_len - min_len <= 1, \
                'misaligned features ([max] {:d}, [min] {:d}) for video {:s}' \
                ''.format(max_len, min_len, vid_id)

            # pad shorter sequences by replicating last feature vector
            for idx in range(len(vid_feats)):
                if feat_lens[idx] < max_len:
                    pad = np.tile(vid_feats[idx][-1], (max_len - feat_lens[idx], 1))
                    vid_feats[idx] = np.concatenate((vid_feats[idx], pad))

            # concatenate features along channel dimension
            vid_feats = np.concatenate(vid_feats, axis=-1)  # (t, c)
        else:
            vid_feats = vid_feats[0]

        # temporally down-sample features
        if self.downsample_rate > 1:
            vid_feats = vid_feats[::self.downsample_rate]

        vid_feats = vid_feats.transpose()                   # (c, t)
        vid_feats = torch.from_numpy(np.ascontiguousarray(vid_feats))

        # normalize features to unit length
        if self.normalize_vid:
            vid_feats = F.normalize(vid_feats, dim=0)
        return vid_feats

    def _truncate_vid_feats(
        self,
        feats,          # float tensor (c, t), full video features 
        segments,       # float tensor (n, 2), event segments
        offset,         # float, clip offset
        num_trials=5000 # int, number of trials
    ):
        vid_len = feats.size(1)
        max_vid_len = self.max_vid_len

        if vid_len <= max_vid_len:
            if self.crop_ratio is None:
                return feats, segments

            max_vid_len = random.randint(
                max(np.ceil(self.crop_ratio[0] * vid_len), 1),
                min(np.ceil(self.crop_ratio[1] * vid_len), vid_len)
            )
            if max_vid_len == vid_len:
                return feats, segments

        # rough estimate on the range of valid chunks
        s0 = max(0, np.floor(segments[:, 0].max() - max_vid_len))
        s1 = min(vid_len - max_vid_len, np.ceil(segments[:, 1].min()))
        
        seg_lens = torch.clamp(segments[:, 1] - segments[:, 0], min=1e-5)

        for _ in range(num_trials):
            ws = random.randint(s0, s1) # window start
            we = ws + max_vid_len       # window end

            # check overlap with segments
            start = torch.clamp(segments[:, 0], min=ws - offset)
            end = torch.clamp(segments[:, 1], max=we + offset)
            overlap = torch.clamp(end - start, min=0)
            if torch.all(overlap / seg_lens > self.trunc_thresh):
                feats = feats[:, ws:we]
                segments = torch.clamp(
                    segments - ws, min=-offset, max=we - ws + offset
                )
                return feats, segments

        raise ValueError('no valid truncation found')

    def _load_text_feats(self, text_id):
        if self.tokenizer is not None:
            text_feats = self.tokenizer(self.text_dict[text_id]['text'])
        else:
            try:
                text_feat_file = os.path.join(self.text_feat_dir, text_id + '.npy')
                text_feats = np.load(text_feat_file).astype(np.float32)
            except:
                raise ValueError(
                    'failed to load features for sentence {:s}'.format(text_id)
                )
            text_feats = text_feats.transpose()     # (c, t)
            text_feats = torch.from_numpy(np.ascontiguousarray(text_feats))
            
        if self.is_training:
            text_feats = text_feats[:, :self.max_text_len]

        # normalize text features to unit length
        if self.normalize_text:
            text_feats = F.normalize(text_feats, dim=0)

        return text_feats

    def _load_ext_scores(self, text_id):
        try:
            score_file = os.path.join(self.ext_score_dir, text_id + '.npy')
            scores = np.load(score_file).astype(np.float32)
        except:
            raise ValueError(
                'failed to load external scores for sentence {:s}'.format(text_id)
            )

        # temporally down-sample scores
        if self.downsample_rate > 1:
            scores = scores[::self.downsample_rate]

        scores = torch.from_numpy(np.ascontiguousarray(scores))[None]   # (1, t)

        if self.normalize_scores:
            scores = torch.sigmoid(scores / self.temperature)

        return scores

    def _avgpool_to_fixed_len(self, feats, size):
        vid_len = feats.size(1)
        sampling_ratio = math.ceil(vid_len / size)
        feats = F.interpolate(
            feats[None],
            size=size * sampling_ratio, mode='linear', align_corners=False
        )
        if sampling_ratio > 1:
            feats = F.avg_pool1d(feats, kernel_size=sampling_ratio)
        feats = feats[0]

        return feats

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


@register_dataset('video_centric')
class VideoCentricDataset(BaseDataset):
    """
    Dataset for video grounding where a training sample is defined by a
    video and a subset of its associated text queries.

    NOTE: this class behaves correctly in training only when all processes
    share exactly the same random seed during initialization. This allows
    identical grouping of training samples across all processes.

    Expected behavior:
    - train: a video + no more than max_num_text text queries
    - eval: a video + all of its text queries
    """
    def __init__(self, **kwargs):
        super(VideoCentricDataset, self).__init__(**kwargs)

        if self.is_training:
            self.data_list = self._build_train_samples()
        else:
            assert self.num_epochs == 1
            self.data_list = self._build_eval_samples()

    def _build_train_samples(self):
        samples = []
        ## NOTE: here we pre-calculated samples for all epochs
        ## we do not shuffle samples here and rely on sampler to do it
        for _ in range(self.num_epochs):
            for vid_id in self.vid_dict.keys():
                samples += self._group(vid_id)
        samples = samples[:len(samples) // self.num_epochs * self.num_epochs]
        return tuple(samples)

    def _build_eval_samples(self):
        samples = []
        for vid_id, vid_dict in self.vid_dict.items():
            samples += [(vid_id, tuple(range(len(vid_dict['segments']))))]
        return tuple(samples)

    def _group(self, vid_id):
        if self.to_fixed_len:
            return self._group_with_fixed_len(vid_id)
        return self._group_with_max_len(vid_id)

    def _group_with_fixed_len(self, vid_id):
        vid_dict = self.vid_dict[vid_id]
        idx = list(range(len(vid_dict['segments'])))

        if self.group_method in ("random", "all"):
            return [(vid_id, tuple(idx))]

        random.shuffle(idx)
        samples = []
        for i in range(0, len(idx), self.max_num_text):
            sample = (vid_id, tuple(idx[i:i + self.max_num_text]))
            samples += [sample]
        return samples

    def _group_with_max_len(self, vid_id):
        vid_dict = self.vid_dict[vid_id]

        # worse-case window size
        if vid_dict['num_clips'] <= self.max_vid_len:
            win_len = vid_dict['num_clips']
            if self.crop_ratio is not None:
                win_len = max(np.ceil(self.crop_ratio[0] * win_len), 1)
        else:
            win_len = self.max_vid_len
        win_len = (
            self.clip_stride * (win_len - 1) + self.clip_size
        ) / vid_dict['fps'] # window length in seconds

        # sort segments in ascending order of start time
        sort_idx = np.argsort(vid_dict['segments'][:, 0])
        segments = vid_dict['segments'][sort_idx]
        mask = np.ones(len(segments), dtype=bool)

        samples = []
        while mask.sum() > 0:
            # probe selection
            ## NOTE: our heuristic is to always select 1st available segment
            ptr = np.nonzero(mask)[0].min()

            # largest window covering probe
            ## NOTE: here we do not consider truncation effect for simplicity
            ## this also adds some room for temporal jittering in data loading
            ws, we = segments[ptr, 0], segments[ptr, 0] + win_len
            if segments[ptr, 1] - segments[ptr, 0] > win_len:
                idx = np.array([ptr])   # corner case: segment longer than window
            else:
                is_inside = (
                    (segments[:, 0] >= ws) & (segments[:, 1] <= we) & mask
                )   # candidates fully covered by window
                idx = np.nonzero(is_inside)[0]
                if len(idx) > self.max_num_text:
                    # sample a subset if too many candidates
                    idx = np.random.choice(idx, self.max_num_text, replace=False)
            sample = (vid_id, tuple(sort_idx[idx]))
            samples += [sample]
            mask[idx] = 0
        return samples

    def __len__(self):
        return len(self.data_list) // self.num_epochs

    def __getitem__(self, idx):
        vid_id, seg_idx = self.data_list[self.epoch * len(self) + idx]
        vid_dict = self.vid_dict[vid_id]

        # load video features (c, t)
        vid_feats = self._load_vid_feats(vid_id)
        vid_len = vid_feats.size(1)

        # resize video features and update clip stride / size
        clip_size, clip_stride = self.clip_size, self.clip_stride
        if self.to_fixed_len:
            vid_feats = self._avgpool_to_fixed_len(vid_feats, self.max_vid_len)
            clip_size = clip_stride = float(
                ((vid_len - 1) * clip_stride + clip_size) / self.max_vid_len
            )
        clip_offset = 0.5 * clip_size / clip_stride

        # locate timestamps in temporal feature grid
        ## NOTE: center feature around the middle frame of the clip
        segments = np.clip(
            vid_dict['segments'][np.array(seg_idx)] * vid_dict['fps'], 
            a_min=0, a_max=vid_dict['num_frames']
        ) / clip_stride - clip_offset
        segments = torch.from_numpy(
            np.ascontiguousarray(segments.astype(np.float32))
        )

        # truncate video features and update target segments
        if self.is_training:
            if not self.to_fixed_len:
                vid_feats, segments = self._truncate_vid_feats(
                    vid_feats, segments, clip_offset
                )
            if self.group_method == "random" and len(seg_idx) > self.max_num_text:
                seg_idx = random.sample(seg_idx, k=self.max_num_text)
                segments = segments[seg_idx]

        # load text features / IDs
        text_feats_list = tuple()
        for idx in seg_idx:
            text_feats = self._load_text_feats(vid_dict['text_ids'][idx])
            text_feats_list += (text_feats, )

        # load external scores (only for inference)
        if not self.is_training and self.ext_score_dir is not None:
            ext_scores_list = tuple()
            for idx in seg_idx:
                scores = self._load_ext_scores(vid_dict['text_ids'][idx])
                if self.to_fixed_len:
                    scores = self._avgpool_to_fixed_len(
                        scores, self.max_vid_len
                    )
                ext_scores_list += (scores, )
            ext_scores = torch.cat(ext_scores_list)
        else:
            ext_scores = None

        return {
                 'fps'        : vid_dict['fps'],        # frames per second
                 'num_frames' : vid_dict['num_frames'], # total number of frames
                 'duration'   : vid_dict['duration'],   # video duration in seconds
                 'segment'    : vid_dict['segments'],   # ground-truth segments in seconds
                 'clip_size'  : clip_size,              # number of frames per clip
                 'clip_stride': clip_stride,            # effective clip stride
                 'target'     : segments,               # event segments in grid unit

                 'vid'        : vid_feats,              # video features (c2, t2)
                 'text'       : text_feats_list,        # text features List[(c1, t1) x n]
                 'ext_scores' : ext_scores,             # external scores (n, t2)
                }


@register_dataset('text_centric')
class TextCentricDataset(BaseDataset):
    """
    Dataset for video grounding where a training sample is defined by a
    video-text pair (where the text serves as the probe) and optionally 
    includes addition text queries from the same video. The dataset size 
    is equal to the total number of text queries from all videos.

    Expected behavior:
    - train: a video + a single text query
    - eval: a video + a single text query
    """
    def __init__(self, **kwargs):
        super(TextCentricDataset, self).__init__(**kwargs)

        self.data_list = tuple(self.text_dict.keys())

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        text_id = self.data_list[idx]
        text_dict = self.text_dict[text_id]
        vid_id = text_dict['vid_id']
        vid_dict = self.vid_dict[vid_id]

        # load video features (c, t)
        vid_feats = self._load_vid_feats(vid_id)
        vid_len = vid_feats.size(1)

        # resize video features and update clip stride / size
        clip_size, clip_stride = self.clip_size, self.clip_stride
        if self.to_fixed_len:
            vid_feats = self._avgpool_to_fixed_len(vid_feats, self.max_vid_len)
            clip_size = clip_stride = float(
                ((vid_len - 1) * clip_stride + clip_size) / self.max_vid_len
            )
        clip_offset = 0.5 * clip_size / clip_stride

        # locate timestamps in temporal feature grid
        ## NOTE: center feature around the middle frame of the clip
        segments = np.clip(
            text_dict['segment'] * vid_dict['fps'], 
            a_min=0, a_max=vid_dict['num_frames']
        ) / clip_stride - clip_offset
        segments = torch.from_numpy(
            np.ascontiguousarray(segments.astype(np.float32))
        )

        # truncate video features and update target segments
        ## NOTE: use current text as probe
        if self.is_training and not self.to_fixed_len:
            vid_feats, segments = self._truncate_vid_feats(
                vid_feats, segments, clip_offset
            )

        # load text features
        text_feats = self._load_text_feats(text_id)

        # load external scores (only for inference)
        if not self.is_training and self.ext_score_dir is not None:
            ext_scores = self._load_ext_scores(text_id)
            if self.to_fixed_len:
                ext_scores = self._avgpool_to_fixed_len(
                    ext_scores, self.max_vid_len
                )
            ext_scores = ext_scores[0]
        else:
            ext_scores = None
        
        return {
                 'fps'        : vid_dict['fps'],        # frames per second
                 'num_frames' : vid_dict['num_frames'], # total number of frames
                 'duration'   : vid_dict['duration'],   # video duration in seconds
                 'segment'    : text_dict['segment'],   # ground-truth segments in seconds
                 'clip_size'  : clip_size,              # number of frames per clip
                 'clip_stride': clip_stride,            # effective clip stride
                 'target'     : segments,               # event segment in grid unit

                 'vid'        : vid_feats,              # video features (c2, t2)
                 'text'       : text_feats,             # text features (c1, t1)
                 'ext_scores' : ext_scores,             # external scores (t2, )
                }


def make_dataset(opt, num_epochs=1, is_training=True):
    opt = deepcopy(opt)
    if 'tokenizer' in opt:
        tokenizer = make_tokenizer(opt.pop('tokenizer'))
    else:
        tokenizer = None
    return datasets[opt.pop('name')](
        tokenizer=tokenizer, is_training=is_training, num_epochs=num_epochs,  **opt
    )


def make_dataloader(
    dataset,            # dataset
    generator,          # random number generator that controls worker seed
    batch_size,         # local batch size
    num_workers,        # local number of workers
    is_training,        # whether is in training
    world_size=1,       # number of processes (GPUs)
    rank=0,             # current process
):
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=is_training)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=partial(worker_init_reset_seed, num_workers, rank),
        sampler=sampler,
        shuffle=(sampler is None and is_training),
        drop_last=is_training,
        generator=generator,
        persistent_workers=True if num_workers > 0 else False,
    )
    return loader, sampler