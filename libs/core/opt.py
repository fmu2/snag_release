from copy import deepcopy
import math
import yaml


DEFAULTS = {
    'seed': 1234567891,

    'model': {
        'text_net': {
            'name': 'transformer',
            'in_dim': 300,
            'embd_dim': 128,
            'max_seq_len': 24,
        },

        'vid_net': {
            'name': 'transformer',
            'in_dim': 500,
            'embd_dim': 128,
            'n_heads': 4,
            'max_seq_len': 256,
            'stride': 1,
            'arch': (2, 0, 7),
            'mha_win_size': 5,
            'attn_pdrop': 0.0,
            'proj_pdrop': 0.1,
            'path_pdrop': 0.1,
            'use_abs_pe': True,
        },

        'fusion': {
            'name': 'xattn',
            'n_layers': 2,
            'n_heads': 4,
            'attn_pdrop': 0.0,
            'proj_pdrop': 0.1,
            'path_pdrop': 0.1,
            'xattn_mode': 'adaln',
        },

        'cls_head': {
            'name': 'cls',
            'n_layers': 2,
            'prior_prob': 0.0, 
        },

        'reg_head': {
            'name': 'reg',
            'n_layers': 2,
        },
    },

    'pt_gen': {
        'regression_range': 4,
        'sigma': 0.5,
    },

    'train': {
        'data': {
            'split': 'train',
            'downsample_rate': 1,
            'trunc_thresh': 0.5,
            'crop_ratio': (0.9, 1.0),
        },

        'batch_size': 16,
        'num_workers': 4,

        'epochs': 5,
        'warmup_epochs': 5,
        'ema_beta': 0.999,

        'center_sampling': 'radius',
        'center_sampling_radius': 1.5,

        'loss_norm': 160,
        'loss_norm_momentum': 0.9,
        'loss_weight': 1.0,
        'reg_loss': 'diou',

        'optimizer': {
            'name': 'adamw',
            'lr': 1e-3,
            'weight_decay': 0.05,
        },
        'clip_grad_norm': 1.0,

        'scheduler': {
            'name': 'multistep',
            'steps': (-1, ),
            'gamma': 0.1,
        },
    },

    'eval': {
        'data': {
            'split': 'test',
        },

        'ranks': (1, 5),
        'iou_threshs': (0.3, 0.5),
        
        'pre_nms_thresh': 0.001,
        'pre_nms_topk': 2000,
        'seg_len_thresh': 0.1,

        'nms': {
            'mode': 'soft_nms',
            'iou_thresh': 0.1,
            'min_score': 0.001,
            'max_num_segs': 5,
            'sigma': 0.9,
            'voting_thresh': 0.95,
        },
    },

    'log': {
        'log_interval': 100,
        'checkpoint_epochs': (6, 7, 8, 9, 10),
    },
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def _update_opt(opt, is_training=True):
    max_text_len = opt['model']['max_text_len'] = opt['model']['text_net']['max_seq_len']
    max_vid_len = opt['model']['max_vid_len'] = opt['model']['vid_net']['max_seq_len']
    vid_stride = opt['model']['vid_stride'] = opt['model']['vid_net']['stride']
    num_fpn_levels = opt['model']['num_fpn_levels'] = opt['model']['vid_net']['arch'][-1]
    opt['model']['mha_win_size'] = opt['model']['vid_net']['mha_win_size']
    
    opt['train']['data']['max_text_len'] = max_text_len
    opt['train']['data']['max_vid_len'] = vid_stride * max_vid_len
    opt['train']['scheduler']['epochs'] = opt['train']['epochs']
    opt['train']['scheduler']['warmup_epochs'] = opt['train']['warmup_epochs']
    if not is_training:
        eval_data_opt = deepcopy(opt['train']['data'])
        eval_data_opt['name'] = opt['eval']['data']['name']
        eval_data_opt['split'] = opt['eval']['data']['split']
        opt['eval']['data'] = eval_data_opt

    text_dim = opt['model']['text_net']['embd_dim']
    vid_dim = opt['model']['vid_net']['embd_dim']
    opt['model']['fusion']['text_dim'] = text_dim
    opt['model']['fusion']['vid_dim'] = vid_dim
    opt['model']['cls_head']['embd_dim'] = vid_dim
    opt['model']['reg_head']['embd_dim'] = vid_dim
    opt['model']['reg_head']['num_fpn_levels'] = num_fpn_levels

    # derive point generator parameters
    ## NOTE: buffer more points for longer sequence at inference time
    n = 1
    if not is_training:
        n = math.ceil(
            opt['eval'].get('max_vid_len', max_vid_len * 4) / max_vid_len
        )
    opt['pt_gen']['num_fpn_levels'] = num_fpn_levels
    opt['pt_gen']['max_seq_len'] = max_vid_len * n


def load_opt(filepath, is_training=True):
    with open(filepath, 'r') as fd:
        opt = yaml.load(fd, Loader=yaml.FullLoader)
    
    _merge(DEFAULTS, opt)
    _update_opt(opt, is_training)
    return opt