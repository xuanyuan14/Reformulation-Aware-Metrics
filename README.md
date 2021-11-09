# Reformulation-Aware-Metrics

[![THUIR](https://img.shields.io/badge/THUIR-ver%201.0-blueviolet)](www.thuir.cn)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![code-size](https://img.shields.io/github/languages/code-size/xuanyuan14/Reformulation-Aware-Metrics?color=green)]()

## Introduction
This codebase contains source-code of the Python-based implementation of our CIKM 2021 paper.
  - [Chen, Jia, et al. "Incorporating Query Reformulating Behavior into Web Search Evaluation." To Appear in Proceedings of the 30th ACM International Conference on Information and Knowledge Management. 2021.](http://www.thuir.cn/group/~YQLiu/publications/CIKM2021Chen.pdf)


## Requirements
* python 2.7
* sklearn
* scipy

## Data Preparation
Preprocess two datasets [**TianGong-SS-FSD**](http://www.thuir.cn/tiangong-ss-fsd/) and [**TianGong-Qref**](http://www.thuir.cn/tiangong-qref/) into the the following format:
```
[Reformulation Type]<tab>[Click List]<tab>[Usefulness List]<tab>[Satisfaction Label]
```
* ```Reformulation Type```: A (Add), D (Delete), K (Keep), T (Transform or Change), O (Others), F (First Query). 
* ```Click List```: 1 -- Clicked, 0 -- Not Clicked. 
* ```Usefulness List```: Usefulness or Relevance, 4-scale in TianGong-QRef, 5-scale in TianGong-SS-FSD.  
* ```Satisfaction Label```: 5-scale for both datasets.  

Then, bootsrap them into N samples and put the bootstapped data (directories) into ```./data/bootstrap_fsd``` and ```./data/bootstrap_qref```.

## Results
The results for each metrics are shown in the following table:

<!-- | Datasets <td colspan=3>TianGong-Qref  <td colspan=2>TianGong-SS-FSD --> 
<!-- |         | TianGong-Qref | TianGong-SS-FSD | -->
| Metric  |   Qref-Spearman |  Qref-Pearson   |   Qref-MSE |  FSD-Spearman |  FSD-Pearson  |  FSD-MSE |
| :---: | :--: | :---: | :---: | :--: | :---: | :---: |
| RBP     |  0.4375 | 0.4180  |  N/A | 0.4898 | 0.5222 | N/A |
| DCG     |  0.4434 | 0.4182  |  N/A | 0.5022 | 0.5290 | N/A | 
| BPM     |  0.4552 | 0.3915  |  N/A | 0.5801 | 0.6052 | N/A |
| RBP sat  |  0.4389 |  0.4170  | N/A | 0.5165 | 0.5527 | N/A |
| DCG sat  |  0.4446 |  0.4166  | N/A | 0.5047 | 0.5344 | N/A |
| BPM sat  |  0.4622 |  0.3674  | N/A | 0.5960 | 0.6029 | N/A |
| rrDBN   |  0.4123 | 0.3670 | 1.1508 | 0.5908 | 0.5602 | 1.0767 |
| rrSDBN  |  0.4177 | 0.3713 | 1.1412 | 0.5991 | 0.5703 | 1.0524 |
| uUBM    |  0.4812 | 0.4303 | 1.0607 | 0.6242 | 0.5775 | 0.8795 |
| uPBM    |  0.4827 | 0.4369 | 1.0524 | 0.6210 | 0.5846 | 0.8644 |
| uSDBN   |  0.4837 | 0.4375 | 1.1443 | 0.6290 | 0.6081 | 0.8840 |
| uDBN    |  0.4928 | 0.4458 | 1.0801 | 0.6339 | 0.6207 | 0.8322 |

To reproduce the results of traditional metrics such as RBP, DCG and BPM, we recommend you to use this repo: [cwl_eval](https://github.com/ireval/cwl). 🤗
 

## Quick Start
To train RAMs, run the script as follows:  
```bash
python run.py --click_model DBN \
	--data qref --id 0 \
	--metric_type expected_utility \
	--max_usefulness 3 \
	--k_num 6 \
	--max_dnum 10 \
	--iter_num 10000 \
	--alpha 0.01 \
	--alpha_decay 0.99 \
	--lamda 0.85 \
	--patience 5 \
	--use_knowledge
```
* ```click_model```: options: ['```DBN```', '```SDBN```', '```UBM```', '```PBM```']
* ```data```: options: ['```fsd```', '```qref```']
* ```metric_type```: options: ['```expected_utility```', '```effort```']
* ```id```: the bootstrapped sample id.
* ```k_num```: the number of user intent shift type will be considered, should be less than or equal to six.
* ```max_dnum```: the maximum number of top documents to be considered for a specific query.
* ```use_knowledge```: whether to use the transition probability from syntactic reformulation types to intent-level ones derived from the TianGong-Qref dataset.

## Citation
If you find the resources in this repo useful, please do not save your star and cite our work:

```bibtex
@inproceedings{chen2021incorporating,
  title={Incorporating Query Reformulating Behavior into Web Search Evaluation},
  author={Chen, Jia and Liu, Yiqun and Mao, Jiaxin and Zhang, Fan and Sakai, Tetsuya and Ma, Weizhi and Zhang, Min and Ma, Shaoping},
  booktitle={Proceedings of the 30th ACM International Conference on Information and Knowledge Management},
  year={2021},
  organization={ACM}
}
```

## Contact
If you have any questions, please feel free to contact me via [chenjia0831@gmail.com]() or open an issue.
