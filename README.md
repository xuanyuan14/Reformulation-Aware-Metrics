# Reformulation-Aware-Metrics

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

## Introduction
This codebase contains source-code of the Pytorch-based implementation of our CIKM 2021 paper.
  - [CIKM 2021] [Incorporating Query Reformulating Behavior into Web Search Evaluation. Chen et al.](http://www.thuir.cn/group/~YQLiu/publications/CIKM2021Chen.pdf)


### Requirements
* python 2.7
* sklearn
* scipy

## Data Preparation
Preprocess two datasets [**TianGong-SS-FSD**](http://www.thuir.cn/tiangong-ss-fsd/) and [**TianGong-Qref**](http://www.thuir.cn/tiangong-qref/) into the the following format:
```
[Reformulation Type]<tab>[Click List]<tab>[Usefulness List]<tab>[Satisfaction Label]
```
```Reformulation Type```: A (Add), D (Delete), K (Keep), T (Transform or Change), O (Others), F (First Query)
```Click List```: 1 -- Clicked, 0 -- Not Clicked
```Usefulness List```: Usefulness or Relevance, 4-scale in TianGong-QRef, 5-scale in TianGong-SS-FSD
```Satisfaction Label```: 5-scale for both datasets 

Then, bootsrap them into N samples and put the bootstapped data (directories) into ```./data/bootstrap_fsd``` and ```./data/bootstrap_qref```.

## Quick Start
To train RAMs, run the script as follows:  
```
python run.py --click_model DBN \
	--data qref \
	--id 0 \
	--metric_type expected_utility \
	--max_usefulness 3 \
	--k_num 6 \
	--max_dnum 10 \
	--iter_num 10000 \
	--alpha 0.01 \
	--alpha_decay 0.99 \
	--lamda 0.85 \
	--patience 5 \
	--use_knowledge True
```


## Citation
If you find the resources in this repo useful, please cite our work:
```
@inproceedings{chen2021incorporating,
  title={Incorporating Query Reformulating Behavior into Web Search Evaluation},
  author={Chen, Jia and Liu, Yiqun and Mao, Jiaxin and Zhang, Fan and Sakai, Tetsuya and Ma, Weizhi and Zhang, Min and Ma, Shaoping},
  booktitle={Proceedings of the 30th ACM International Conference on Information and Knowledge Management},
  year={2021},
  organization={ACM}
}
```
