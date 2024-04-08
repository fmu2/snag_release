import argparse
import os

import torch
from libs import load_opt, Evaluator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help="job name")
    parser.add_argument('--ckpt', type=str, help="checkpoint name")
    args = parser.parse_args()

    root = os.path.join('experiments', args.name)
    try:
        opt = load_opt(os.path.join(root, 'opt.yaml'), is_training=False)
    except:
        raise ValueError('experiment folder not found')
    assert os.path.exists(os.path.join(root, 'models', f'{args.ckpt}.pth'))
    opt['_root'] = root
    opt['_ckpt'] = args.ckpt

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    
    evaluator = Evaluator(opt)
    evaluator.run()