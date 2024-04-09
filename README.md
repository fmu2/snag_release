# SnAG: Scalable and Accurate Video Grounding (*CVPR 2024*)

## Introduction

This code repo implements SnAG, a scalable and accurate model for long-form video grounding --- localizing moments within an untrimmed long video based on text descriptions. SnAG features a minimalist, late-fusion design for scalable inference, while supporting video-centric sampling for scalable training. Without bells and whistles, SnAG achieves 44.86% R1\@0.5 and 70.66% R5\@0.5 on TACoS, outperforming the previous state of the art by 8.53 and 12.75 absolute percentage points, respectively. Further, SnAG demonstrates strong results on Ego4D-NLQ (13.57% mean R1 and 32.92 mean R5) and the more challenging MAD dataset (5.55 R1\@0.5 and 13.75 R5\@0.5). Our paper is accepted to CVPR 2024 and an arXiv version can be found at [this link](https://arxiv.org/abs/2404.02257).

Related projects:
> [**ActionFormer: Localizing Moments of Actions with Transformers**](https://arxiv.org/abs/2202.07925) <br>
> Chenlin Zhang, Jianxin Wu, Yin Li <br>
> *ECCV 2022* <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/happyharrycn/actionformer_release)  [![github](https://img.shields.io/github/stars/happyharrycn/actionformer_release.svg?style=social)](https://github.com/happyharrycn/actionformer_release)  [![arXiv](https://img.shields.io/badge/Arxiv-2202.07925-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2202.07925) <br>

## Changelog
* 04/03/2024: Initial code release.

* 02/26/2024: The paper is accepted to CVPR 2024.

## Code Overview
The structure of this code repo is heavily inspired by ActionFormer. Some of the main components are
* ./libs/core: Parameter configuration module.
* ./libs/data: Data loader and IO module.
* ./libs/modeling: Our main model with all its building blocks.
* ./libs/worker.py: Training and evaluation loops.

## Installation
* Follow INSTALL.md for installing necessary dependencies and compiling the code.

## To Reproduce Our Results on TACoS
**Download Features and Annotations**
* Download *tacos.tar.gz* (`md5sum a96537114a930038ab8ddb64a17df6e0`) from [this Google Drive link](https://drive.google.com/file/d/1T5bd6_Z9zuSFastgZu88Uos1PQf1FPqN/view?usp=drive_link). The file includes C3D features in npy format and annotations in json format.

**Details**: The features are extracted using the C3D model pretrained on Sports1M, given clips of `16 frames` with a frame rate of `~30 fps` and a stride of `4 frames`. This gives one feature vector per `4/30 ~= 0.1333` seconds.

**Unpack Features and Annotations**
* Unpack the file under *./data* (or elsewhere and link to *./data*).
* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───data/
│    └───tacos/
│    │	 └───annotations
│    │	 └───c3d_features   
│    └───...
|
└───libs
│
│   ...
```

**Training and Evaluation**
* Train our SnAG with C3D features. This will create an experiment folder under *./experiments* that stores training config, logs, and checkpoints.
```shell
python ./train.py --opt video_centric/tacos.yaml --name tacos_reproduce
```
* [Optional] Monitor the training using TensorBoard
```shell
tensorboard --logdir=./experiments/tacos_reproduce/tensorboard
```
* Evaluate the trained model. The expected R1\@0.5 and R5\@0.5 should be around 45.0(%) and 70.5(%).
```shell
python ./eval.py --name tacos_reproduce --ckpt last
```
* Training and inference on TACoS requires ~7.4 GB and ~1.2 GB of GPU memory. We recommend using a GPU with at least 8 GB of memory. Please set `microbatch_size` in the config file for gradient accumulation over micro-batches when training with less GPU memory (e.g., set `microbatch_size` to half of `batch_size` for GPUs with 6 GB of memory).

**[Optional] Evaluating Our Pre-trained Model**

We also provide a pre-trained model for TACoS. The model with all training logs can be downloaded from [this Google Drive link](https://drive.google.com/file/d/1sk13S_ubwnXIWItsod45NxGsSuhhoV7w/view?usp=sharing). To evaluate the pre-trained model, please follow the steps listed below.

* Unpack the file under *./experiments* (or elsewhere and link to *./experiments*).
* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───experiments/
│    └───tacos_reproduce/
│    │	 └───eval_last.txt
│    │	 └───log.txt
│    │   └───...    
│    └───...
|
└───libs
│
│   ...
```
* The training config is recorded in *./experiments/tacos_reproduce/opt.yaml*.
* The training log is located at *./experiments/tacos_reproduce/log.txt* and also *./experiments/tacos_reproduce/tensorboard*.
* The pre-trained model is *./experiments/tacos_reproduce/models/last.pth*.
* Evaluate the pre-trained model.
```shell
python ./eval.py --name tacos_reproduce --ckpt last
```

* The results (Recall at tIoUs) should be

| Method            | R1\@0.3 | R1\@0.5 | R5\@0.3 | R5\@0.5 |
|-------------------|:-------:|:-------:|:-------:|:-------:|
| SnAG              |  55.51  |  45.14  |  81.58  |  70.31  |

* Training your own model will yield slightly different results due to randomness, yet the results should be close to what we report in the paper.

## To Reproduce Our Results on Ego4D-NLQ
**Download Features and Annotations**
* Download *ego4d_slowfast_bert.tar.gz* (`md5sum d57d03737493e4c7aae39dd3d3d5597b`) from [this Google Drive link](https://drive.google.com/file/d/1G5yR3VrGYpQdzoj7gcSeKiDrICwpbI6b/view?usp=sharing). The file includes SlowFast and BERT features in npy format and annotations in json format.
* Download *ego4d_egovlp.tar.gz* (`md5sum 44e013aa5c4dcbc4d474fdba5c172804`) from [this Google Drive link](https://drive.google.com/file/d/1EI9FDY45V3VCFpv0yOxdJkSXdj_MTbJY/view?usp=sharing). The file includes EgoVLP video and text features in npy format and annotations in json format.

**Details**: We use the official SlowFast features from [here](https://ego4d-data.org/docs/data/features/). They are extracted using the SlowFast model pretrained on Kinetics 400, given clips of `32 frames` with a frame rate of `30 fps` and a stride of `16 frames`. This gives one feature vector per `16/30 ~= 0.533` seconds. The EgoVLP features are extracted using the [EgoVLP model checkpoint](https://drive.google.com/file/d/1-cP3Gcg0NGDcMZalgJ_615BQdbFIbcj7), given clips of `32 frames` with a frame rate of `30 fps` and a stride of `8 frames`. This gives one feature vector per `8/30 ~=0.267` seconds. In practice, SnAG uses 2x-subsampled EgoVLP features (i.e., the *effective* stride is `16 frames`) for fair comparison with baselines.

**Unpack Features and Annotations**
* Unpack the file under *./data* (or elsewhere and link to *./data*).
* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───data/
│    └───ego4d_slowfast_bert/
│    │	 └───annotations
│    │	 └───slowfast_features
│    │	 └───bert_features
│    └───ego4d_egovlp/
│    │	 └───annotations
│    │	 └───egovlp_features
│    └───...
|
└───libs
│
│   ...
```

**Training and Evaluation**
* Train our SnAG with SlowFast + BERT, or EgoVLP features. This will create an experiment folder under *./experiments* that stores training config, logs, and checkpoints.
```shell
python ./train.py --opt video_centric/ego4d_slowfast_bert.yaml --name ego4d_slowfast_bert_reproduce
```
```shell
python ./train.py --opt video_centric/ego4d_egovlp.yaml --name ego4d_egovlp_reproduce
```
* [Optional] Monitor the training using TensorBoard
```shell
tensorboard --logdir=./experiments/ego4d_slowfast_bert_reproduce/tensorboard
```
```shell
tensorboard --logdir=./experiments/ego4d_egovlp_reproduce/tensorboard
```
* Evaluate the trained model. The expected mean R1 and R5 should be around 8.3(%) and 23.6(%) with SlowFast + BERT features, and 13.5(%) and 33.0(%) with EgoVLP features.
```shell
python ./eval.py --name ego4d_slowfast_bert_reproduce --ckpt last
```
```shell
python ./eval.py --name ego4d_egovlp_reproduce --ckpt last
```
* Training and inference on Ego4D-NLQ requires ~3.0 GB and ~1.8 GB of GPU memory, respectively. We recommend using a GPU with at least 4 GB of memory.

**[Optional] Evaluating Our Pre-trained Model**

We also provide pre-trained models for Ego4D-NLQ. The model using SlowFast + BERT features with all training logs can be downloaded from [this Google Drive link](https://drive.google.com/file/d/18BQjk1wFN0qbcdj7IAI80O-uAbEvPL08/view?usp=sharing). The model using EgoVLP features with all training logs can be downloaded from [this Google Drive link](https://drive.google.com/file/d/1_3o7yOlv4F8fmGlnjalo3f91ME2zFn8l/view?usp=sharing). To evaluate the pre-trained model, please follow the steps listed below.

* Unpack the file under *./experiments* (or elsewhere and link to *./experiments*).
* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───experiments/
│    └───ego4d_showfast_bert_reproduce/
│    │	 └───eval_last.txt
│    │	 └───log.txt
│    │   └───...
│    └───ego4d_egovlp_reproduce/
│    │	 └───eval_last.txt
│    │	 └───log.txt
│    │   └───...  
│    └───...
|
└───libs
│
│   ...
```
* The training config is recorded in *./experiments/ego4d_..._reproduce/opt.yaml*.
* The training log is located at *./experiments/ego4d_..._reproduce/log.txt* and also *./experiments/ego4d_..._reproduce/tensorboard*.
* The pre-trained model is *./experiments/ego4d_..._reproduce/models/last.pth*.
* Evaluate the pre-trained model.
```shell
python ./eval.py --name ego4d_slowfast_bert_reproduce --ckpt last
```
```shell
python ./eval.py --name ego4d_egovlp_reproduce --ckpt last
```

* The results (Recall at tIoUs) should be

| Method                 | R1\@0.3 | R1\@0.5 | mean R1 | R5\@0.3 | R5\@0.5 | mean R5 |
|------------------------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| SnAG (SlowFast + BERT) |   9.75  |   6.40  |   8.08  |  28.10  |  19.47  |  23.79  |
| SnAG (EgoVLP)          |  15.53  |  10.94  |  13.24  |  38.40  |  27.70  |  33.10  |

* Training your own model will yield slightly different results due to randomness, yet the results should be close to what we report in the paper.

## To Reproduce Our Results on MAD
**Download Features and Annotations**
* Download *mad.tar.gz* (`md5sum dd4fc6f8e2297eb10a1c82d405b03658`) from [this Google Drive link](https://drive.google.com/file/d/10jZ5U9XStwM5xD__zhWJToatKzScQys5/view?usp=sharing). The file includes CLIP features in npy format and annotations in json format.

**Details**: We use the official CLIP features from [here](https://github.com/Soldelli/MAD). The features are extracted using CLIP ViT-L/14 with a frame rate of `5 fps`. This gives one feature vector every `0.2` seconds.

**Unpack Features and Annotations**
* Unpack the file under *./data* (or elsewhere and link to *./data*).
* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───data/
│    └───mad/
│    │	 └───annotations
│    │	 └───clip_features
│    └───...
|
└───libs
│
│   ...
```

**Training and Evaluation**
* Train our SnAG with CLIP features. This will create an experiment folder under *./experiments* that stores training config, logs, and checkpoints.
```shell
python ./train.py --opt video_centric/mad.yaml --name mad_reproduce
```
* [Optional] Monitor the training using TensorBoard
```shell
tensorboard --logdir=./experiments/mad_reproduce/tensorboard
```
* Evaluate the trained model. The expected R1\@0.5 and R5\@0.5 should be around 5.5(%) and 13.5(%).
```shell
python ./eval.py --name mad_reproduce --ckpt last
```
* Training and inference on MAD requires ~20 GB and ~2.2 GB of GPU memory. We recommend using a GPU with at least 24 GB of memory. When running out of GPU memory, try increasing `batch_size` and reducing `max_num_text` while keeping their product unchanged (e.g., `batch_size: 8`, `max_num_text: 4`), and experiment with different `microbatch_size` (e.g., `microbatch_size: 2`). This will reduce memory footprint at the cost of increasing training time.

**[Optional] Evaluating Our Pre-trained Model**

We also provide a pre-trained model for MAD. The model with all training logs can be downloaded from [this Google Drive link](https://drive.google.com/file/d/19srcs5iK34IzT74VNwBCTi0fZ-P4tRmR/view?usp=sharing). To evaluate the pre-trained model, please follow the steps listed below.

* Unpack the file under *./experiments* (or elsewhere and link to *./experiments*).
* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───experiments/
│    └───mad_reproduce/
│    │	 └───eval_last.txt
│    │	 └───log.txt
│    │   └───...    
│    └───...
|
└───libs
│
│   ...
```
* The training config is recorded in *./experiments/mad_reproduce/opt.yaml*.
* The training log is located at *./experiments/mad_reproduce/log.txt* and also *./experiments/mad_reproduce/tensorboard*.
* The pre-trained model is *./experiments/mad_reproduce/models/last.pth*.
* Evaluate the pre-trained model.
```shell
python ./eval.py --name mad_reproduce --ckpt last
```

* The results (Recall at tIoUs) should be

| Method            | R1\@0.1 | R1\@0.3 | R1\@0.5 | R5\@0.1 | R5\@0.3 | R5\@0.5 |
|-------------------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| SnAG              |  10.35  |   8.51  |   5.47  |  24.40  |  20.30  |  13.41  |

* Training your own model will yield slightly different results due to randomness, yet the results should be close to what we report in the paper.

## To Reproduce Our Results on Charades-STA
**Download Features and Annotations**
* Download *charades_sta_c3d.tar.gz* (`md5sum 10300461e5f713dffcc038506c73aec7`) from [this Google Drive link](https://drive.google.com/file/d/1SsJ_cEUkOWKlRF0p5IGUnXqgicVOY-Ib/view?usp=sharing). The file includes C3D features in npy format and annotations in json format.
* Download *charades_sta_i3d.tar.gz* (`md5sum 57ad93a548dc5428c284e3fc5852136d`) from [this Google Drive link](https://drive.google.com/file/d/1njAWKq3p82SjKVN_H6JMh3kSv6BWJduI/view?usp=sharing). The file includes I3D features in npy format and annotations in json format.

**Details**: The C3D features are extracted using the C3D model pretrained on Sports1M, given clips of `16 frames` with a frame rate of `24 fps` and a stride of `4 frames`. This gives one feature vector per `4/24 ~= 0.167` seconds. The I3D features are extracted using the I3D model pretrained on Kinetics 400, given clips of `16 frames` with a frame rate of `24 fps` and a stride of `4 frames`. This gives one feature vector per `4/24 ~= 0.167` seconds.

**Unpack Features and Annotations**
* Unpack the file under *./data* (or elsewhere and link to *./data*).
* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───data/
│    └───charades_sta_c3d/
│    │	 └───annotations
│    │	 └───c3d_features
│    └───charades_sta_i3d/
│    │	 └───annotations
│    │	 └───i3d_features
│    │	 |   └───charades   # not used
|    |   |   └───kinetics
│    └───...
|
└───libs
│
│   ...
```

**Training and Evaluation**
* Train our SnAG with C3D or I3D features. This will create an experiment folder under *./experiments* that stores training config, logs, and checkpoints.
```shell
python ./train.py --opt video_centric/charades_sta_c3d.yaml --name charades_sta_c3d_reproduce
```
```shell
python ./train.py --opt video_centric/charades_sta_i3d.yaml --name charades_sta_i3d_reproduce
```
* [Optional] Monitor the training using TensorBoard
```shell
tensorboard --logdir=./experiments/charades_sta_c3d_reproduce/tensorboard
```
```shell
tensorboard --logdir=./experiments/charades_sta_i3d_reproduce/tensorboard
```
* Evaluate the trained model. The expected R1\@0.7 and R5\@0.7 should be around 33.5(%) and 65.5(%) with C3D features, and 46.5(%) and 73.0(%) with I3D features.
```shell
python ./eval.py --name charades_sta_c3d_reproduce --ckpt last
```
```shell
python ./eval.py --name charades_sta_i3d_reproduce --ckpt last
```
* Training and inference on Charades-STA requires ~2.7 GB and ~1.2 GB of GPU memory, respectively. We recommend using a GPU with at least 4 GB of memory.

**[Optional] Evaluating Our Pre-trained Model**

We also provide pre-trained models for Charades-STA. The model using C3D features with all training logs can be downloaded from [this Google Drive link](https://drive.google.com/file/d/1qQNn9iC1zjnPaPfaTCfc-FuL4UNxhj7N/view?usp=sharing). The model using I3D features with all training logs can be downloaded from [this Google Drive link](https://drive.google.com/file/d/123ekEtPCaL87clSr5Mk04NpjRk2chdkG/view?usp=sharing). To evaluate the pre-trained model, please follow the steps listed below.

* Unpack the file under *./experiments* (or elsewhere and link to *./experiments*).
* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───experiments/
│    └───charades_sta_c3d_reproduce/
│    │	 └───eval_last.txt
│    │	 └───log.txt
│    │   └───...
│    └───charades_sta_i3d_reproduce/
│    │	 └───eval_last.txt
│    │	 └───log.txt
│    │   └───...  
│    └───...
|
└───libs
│
│   ...
```
* The training config is recorded in *./experiments/charades_sta_..._reproduce/opt.yaml*.
* The training log is located at *./experiments/charades_sta_..._reproduce/log.txt* and also *./experiments/charades_sta_..._reproduce/tensorboard*.
* The pre-trained model is *./experiments/charades_sta_..._reproduce/models/last.pth*.
* Evaluate the pre-trained model.
```shell
python ./eval.py --name charades_sta_c3d_reproduce --ckpt last
```
```shell
python ./eval.py --name charades_sta_i3d_reproduce --ckpt last
```

* The results (Recall at tIoUs) should be

| Method        | R1\@0.5 | R1\@0.7 | R5\@0.5 | R5\@0.7 |
|---------------|:-------:|:-------:|:-------:|:-------:|
| SnAG (C3D)    |  51.75  |  33.33  |  90.83  |  65.56  |
| SnAG (I3D)    |  65.19  |  46.32  |  93.04  |  73.12  |

* Training your own model will yield slightly different results due to randomness, yet the results should be close to what we report in the paper. With our latest implementation, we achieve +1.5 R5@0.7 compared to the paper results using both C3D and I3D features. The R5@0.5 dropped by 1.5 when using C3D features. We consider this a favorable tradeoff as the predictions become more accurate. 

## To Reproduce Our Results on ActivityNet-Captions
**Download Features and Annotations**
* Download *anet_1.3.tar.gz* (`md5sum 4da7bbd46ebf43906cb44e696a4a1852`) from [this Google Drive link](https://drive.google.com/file/d/18zqt4QoqqBODUFj7JPrvB79vfHeVRpn0/view?usp=sharing). The file includes C3D features in npy format and annotations in json format.

**Details**: We use the official C3D features from [here](http://activity-net.org/challenges/2016/download.html). The features are extracted using the C3D model pretrained on Sports1M, given clips of `16 frames` and a stride of `8 frames`. The frame rate is unknown. The feature dimension has been reduced from 4096 to 500 using PCA.

**Unpack Features and Annotations**
* Unpack the file under *./data* (or elsewhere and link to *./data*).
* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───data/
│    └───anet_1.3/
│    │	 └───annotations
│    │	 └───c3d_features   
│    └───...
|
└───libs
│
│   ...
```

**Training and Evaluation**
* Train our SnAG with C3D features. This will create an experiment folder under *./experiments* that stores training config, logs, and checkpoints.
```shell
python ./train.py --opt video_centric/anet_1.3.yaml --name anet_1.3_reproduce
```
* [Optional] Monitor the training using TensorBoard
```shell
tensorboard --logdir=./experiments/anet_1.3_reproduce/tensorboard
```
* Evaluate the trained model. The expected R1\@0.7 and R5\@0.7 should be around 30.0(%) and 63.0(%).
```shell
python ./eval.py --name anet_1.3_reproduce --ckpt last
```
* Training and inference on ActivityNet-Captions requires ~2.0 GB and ~1.2 GB of GPU memory. We recommend using a GPU with at least 4 GB of memory.

**[Optional] Evaluating Our Pre-trained Model**

We also provide a pre-trained model for ActivityNet-Captions. The model with all training logs can be downloaded from [this Google Drive link](https://drive.google.com/file/d/1bg9SmctG4IVx04L0IH0GJy0K_axgPnRz/view?usp=sharing). To evaluate the pre-trained model, please follow the steps listed below.

* Unpack the file under *./experiments* (or elsewhere and link to *./experiments*).
* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───experiments/
│    └───anet_1.3_reproduce/
│    │	 └───eval_last.txt
│    │	 └───log.txt
│    │   └───...    
│    └───...
|
└───libs
│
│   ...
```
* The training config is recorded in *./experiments/anet_1.3_reproduce/opt.yaml*.
* The training log is located at *./experiments/anet_1.3_reproduce/log.txt* and also *./experiments/anet_1.3_reproduce/tensorboard*.
* The pre-trained model is *./experiments/anet_1.3_reproduce/models/last.pth*.
* Evaluate the pre-trained model.
```shell
python ./eval.py --name anet_1.3_reproduce --ckpt last
```

* The results (Recall at tIoUs) should be

| Method            | R1\@0.5 | R1\@0.7 | R5\@0.5 | R5\@0.7 |
|-------------------|:-------:|:-------:|:-------:|:-------:|
| SnAG              |  47.44  |  29.89  |  82.60  |  63.29  |

* Training your own model will yield slightly different results due to randomness, yet the results should be close to what we report in the paper.

## Contact
Fangzhou Mu (fmu2@wisc.edu)

## Reference
If you are using our code, please consider citing our paper.
```
@inproceedings{mu2024snag,
  title={{SnAG}: Scalable and Accurate Video Grounding},
  author={Mu, Fangzhou and Mo, Sicheng and Li, Yin},
  booktitle={CVPR},
  year={2024}
}
```