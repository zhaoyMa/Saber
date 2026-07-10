# Saber: An Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for Diffusion Language Model
[![arXiv](https://img.shields.io/badge/arXiv-2510.18165-b31b1b.svg)](https://arxiv.org/abs/2510.18165)

Our work introduces Saber, a training-free sampling algorithm for diffusion language models that enhances code generation by adaptively accelerating inference and incorporating backtracking, thereby improving output quality and speed while narrowing the performance gap with autoregressive models.

🎉 Saber has won an SAC Highlight Award at ACL'26.
🎉 Our work has been accepted to ACL'26 (main) 


## Installation environment
```shell
conda create -n saber python=3.11
pip install -r requirements.txt
```
or
```shell
uv venv saber --python 3.11
source saber/bin/activate
uv pip install -r requirements.txt
```

## Setup

### 1. Download Model
Download the LLaDA-8B-Instruct model and place it in the `models/` directory.
You can download the model from Hugging Face.

### 2. Prepare Datasets
We provide the HumanEval, MBPP, HumanEval-ET, and MBPP-ET datasets in the `data/` directory. For LiveCodeBench, you will need to download it separately and place it in `data/livecodebench/`.

## Evaluation of Saber
Firstly, ensure the model is placed correctly in the `models/` directory (the configs use the path `./models/LLaDA-8B-Instruct` by default).
Secondly, verify the datasets are in the correct path.
Finally, execute the following command to evaluate:
```shell
python eval.py --config ./configs/humaneval.yaml
```
If you want to test other methods, change the method in the yaml file.
For the humaneval and MBPP datasets, our code will print pass@1 And steps. For the livecodebench dataset, our code will save the generated results, and you need to run the evaluation program yourself

## Citation
```
@article{dong2025saber,
  title={Saber: An Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for Diffusion Language Model},
  author={Yihong Dong and Zhaoyu Ma and Xue Jiang and Zhiyuan Fan and Jiaru Qian and Yongmin Li and Jianha Xiao and Zhi Jin and Rongyu Cao and Binhua Li and Fei Huang and Yongbin Li and Ge Li},
  journal={arXiv preprint arXiv:2510.18165},
  year={2025}
}
```
