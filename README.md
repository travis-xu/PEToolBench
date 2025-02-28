# PEToolBench
Code for our paper [PEToolLLM: Towards Personalized Tool Learning in Large Language Models](https://arxiv.org/abs/2502.18980).

![intro](/assets/fig_intro.png)

## Installation

```
pip install -r requirements.txt
```

Install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

## Dataset

Download the dataset from [HuggingFace](https://huggingface.co/datasets/travisxu/PEToolBench). Then put the training files into /dataset_sft and /dataset_dpo, test files into /dataset_test.

## Test

```
bash scripts/test.sh
```

## Training: PEToolLLaMA

### Stage 1: Personalized SFT

```
bash scripts/train_sft.sh
```

### Stage 2: Personalized DPO

```
bash scripts/train_sft-dpo.sh
```

## Citation

```
@misc{xu2025petoolllmpersonalizedtoollearning,
      title={PEToolLLM: Towards Personalized Tool Learning in Large Language Models}, 
      author={Qiancheng Xu and Yongqi Li and Heming Xia and Fan Liu and Min Yang and Wenjie Li},
      year={2025},
      eprint={2502.18980},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.18980}, 
}
```
