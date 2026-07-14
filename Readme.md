# Saber: An Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for Diffusion Language Model
[![arXiv](https://img.shields.io/badge/arXiv-2510.18165-b31b1b.svg)](https://arxiv.org/abs/2510.18165)

Our work introduces Saber, a training-free sampling algorithm for diffusion language models that enhances code generation by adaptively accelerating inference and incorporating backtracking, thereby improving output quality and speed while narrowing the performance gap with autoregressive models.

🎉 Saber has won an SAC Highlight Award at ACL'26.  
🎉 Our work has been accepted to ACL'26 (main) 

## Environment Requirements

- Python 3.11+
- PyTorch 2.1+
- CUDA 12.1+ (for GPU acceleration)
- GPU with at least 24GB VRAM (recommended for LLaDA-8B-Instruct)

## Installation

```shell
conda create -n saber python=3.11
conda activate saber
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

```shell
mkdir -p models
cd models
git lfs install
git clone https://huggingface.co/your-organization/LLaDA-8B-Instruct
cd ..
```

### 2. Prepare Datasets

We provide the HumanEval, MBPP, HumanEval-ET, and MBPP-ET datasets in the `data/` directory.

For LiveCodeBench, you need to download it separately:

```shell
mkdir -p data/livecodebench
# Download LiveCodeBench test set to data/livecodebench/test.jsonl
```

## Evaluation

### Basic Usage

```shell
python eval.py --config ./configs/humaneval.yaml
```

### Available Configurations

- `configs/humaneval.yaml` - HumanEval dataset
- `configs/mbpp.yaml` - MBPP dataset
- `configs/livecodebench.yaml` - LiveCodeBench dataset

### Switching Methods

Change the `method` field in the YAML config file:

```yaml
method: 'saber'  # Options: saber, default, wino, fast, entropy, margin, remdm
```

### Running with Different Methods

```shell
# Evaluate with Saber on HumanEval
python eval.py --config ./configs/humaneval.yaml

# Evaluate with default method on MBPP
python eval.py --config ./configs/mbpp.yaml  # change method in yaml

# Evaluate with remdm on LiveCodeBench
python eval.py --config ./configs/livecodebench.yaml  # change method in yaml
```

## Configuration Reference

### Top-level Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `dataset_name` | Name of the dataset to evaluate | `humaneval`, `mbpp`, `livecodebench` |
| `model_path` | Path to the LLaDA model | `./models/LLaDA-8B-Instruct` |
| `data_root` | Root directory for datasets | `./data` |
| `method` | Sampling method to use | `saber`, `default`, `wino`, `fast`, `entropy`, `margin`, `remdm` |

### dataset_config Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `load_dataset_args` | Arguments for loading the dataset | `{ path: "humaneval" }` |
| `doc_to_text_fn` | Function to convert document to text prompt | `humaneval_doc_to_text` |
| `extract_answer_fn` | Function to extract answer from generated text | `humaneval_extract_answer` |
| `problem_file` | Path to the problem file for evaluation | `./data/humaneval/HumanEval.jsonl` |
| `problem_file_et` | Path to the ET problem file (optional) | `./data/humaneval/HumanEval_ET.jsonl` |

### generation_args Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gen_length` | Length of generated text | `256` |
| `block_length` | Block length for remasking | `256` |
| `temperature` | Sampling temperature | `0.0` |
| `num_samples` | Number of samples per problem | `1` |

### method_args Parameters

Method-specific arguments can be provided under `method_args`:

```yaml
method: 'remdm'
method_args:
  remdm:
    init_unmask_ratio: 0.875
    unmask_k: 1
    loop_steps: 32
    temperature: 0.0
    cfg_scale: 0.0
    remasking: 'low_confidence'
    mask_id: 126336
```

## Adding a New Dataset

Adding a new dataset is straightforward with our plugin system. Follow these steps:

### Step 1: Create a Dataset Handler

Create a new file in `dataset_utils/` (e.g., `dataset_utils/my_dataset.py`):

```python
from .registry import register_dataset, DatasetHandler

@register_dataset('my_dataset')
class MyDatasetHandler(DatasetHandler):
    def load_dataset(self, config):
        """Load the dataset"""
        from datasets import load_dataset
        dataset_cfg = config['dataset_config']
        dataset_path = os.path.join(config['data_root'], dataset_cfg['load_dataset_args']['path'])
        return load_dataset(dataset_path, split="test")

    def doc_to_text(self, doc):
        """Convert document to model input"""
        prompt = f"Your prompt template here: {doc['question']}"
        context = [{"role": "user", "content": prompt}]
        trailing_prompt = ""
        return context, doc, trailing_prompt

    def extract_answer(self, generated_text, doc):
        """Extract answer from generated text"""
        return generated_text.strip()

    def evaluate(self, outputs, config):
        """Evaluate the results (optional)"""
        return None
```

### Step 2: Create a Config File

Create a new YAML config file in `configs/` (e.g., `configs/my_dataset.yaml`):

```yaml
dataset_name: 'my_dataset'
model_path: "./models/LLaDA-8B-Instruct"
data_root: "./data"

dataset_config:
  load_dataset_args: { path: "my_dataset/test.jsonl" }
  doc_to_text_fn: "my_dataset_doc_to_text"
  extract_answer_fn: "my_dataset_extract_answer"

generation_args: { gen_length: 256, block_length: 256, temperature: 0.0 }
method: 'saber'
```

### Step 3: Run Evaluation

```shell
python eval.py --config ./configs/my_dataset.yaml
```

## Dataset Handler Interface

The `DatasetHandler` base class defines the following methods:

| Method | Description | Required |
|--------|-------------|----------|
| `load_dataset(config)` | Load the dataset from disk | Yes |
| `doc_to_text(doc)` | Convert a document to model input prompt | Yes |
| `extract_answer(text, doc)` | Extract the answer from generated text | Yes |
| `evaluate(outputs, config)` | Evaluate results and return metrics | No |
| `preprocess_generated_text(text, doc)` | Preprocess text before extraction | No |

## Results

### HumanEval and MBPP

For HumanEval and MBPP datasets, the code will print:
- Pass@1 score
- ET (Extended Tests) evaluation results
- Average steps per sample
- Number of samples per problem

### LiveCodeBench

For LiveCodeBench, the generated results will be saved to `./results/` directory. You need to run the LiveCodeBench evaluation script separately.

## Citation

```
@article{dong2025saber,
  title={Saber: An Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for Diffusion Language Model},
  author={Yihong Dong and Zhaoyu Ma and Xue Jiang and Zhiyuan Fan and Jiaru Qian and Yongmin Li and Jianha Xiao and Zhi Jin and Rongyu Cao and Binhua Li and Fei Huang and Yongbin Li and Ge Li},
  journal={arXiv preprint arXiv:2510.18165},
  year={2025}
}
```