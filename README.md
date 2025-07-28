# Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search

[![arXiv](https://img.shields.io/badge/arXiv-2506.11155-b31b1b.svg)](https://aclanthology.org/2025.acl-long.323.pdf)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AutoCaption-orange)](https://huggingface.co/datasets/HasuerYu/AutoCaption)

## 📌 Overview

**AutoCaption** is a novel framework that employs Monte Carlo Tree Search (MCTS) to generate rich, diverse, and detailed video captions. The framework iteratively constructs high-quality video descriptions that thoroughly cover objects, actions, environments, and temporal dynamics.

**MCTS-VCB** is a fine-grained video captioning benchmark automatically constructed using AutoCaption, enabling comprehensive evaluation of Multimodal Large Language Models (MLLMs) on video understanding tasks.

## 🚀 Highlights

- **🧠 AutoCaption Framework**: Iteratively constructs high-quality video descriptions using MCTS, covering objects, actions, environments, and more
- **📊 MCTS-VCB Benchmark**: Contains diverse, multi-faceted video captions for robust MLLM evaluation  
- **🔍 Comprehensive Evaluation**: Benchmarked over 20 MLLMs with Gemini-1.5-Pro achieving the top F1 score of 71.2%
- **📈 Fine-tuning Results**: InternVL2.5-8B fine-tuned on AutoCaption data achieved:
  - **+25.0%** improvement on MCTS-VCB
  - **+16.3%** improvement on DREAM-1K

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ GPU memory for Qwen2-VL-7B

### Quick Install
```bash
git clone https://github.com/your-username/autocaption.git
cd autocaption
pip install -r requirements.txt
```

### Development Install
```bash
git clone https://github.com/your-username/autocaption.git
cd autocaption
pip install -e .
```

### Optional Dependencies
```bash
# For distributed processing
pip install mpi4py

# For experiment tracking
pip install wandb
```

## 🚀 Quick Start

### 1. Prepare Data
Create your input file in JSONL format:
```json
{"video_name": "video1.mp4", "video_path": "/path/to/video1.mp4", "index": 0}
{"video_name": "video2.mp4", "video_path": "/path/to/video2.mp4", "index": 1}
```

### 2. Configure Settings
```bash
# Copy and modify configuration
cp config/config.yaml config/my_config.yaml
# Edit config/my_config.yaml as needed
```

### 3. Run AutoCaption
```bash
# Multi-GPU processing
python main.py \
    --input_path data/videos.jsonl \
    --output_path results/captions.jsonl \
    --process_num 4 \
    --gpu_nums_one_process 2 \
    --max_rollout_times 25 \
    --log_level INFO
```

## 📂 Repository Structure

```
autocaption/
├── 📁 generator/              # Model generators
│   └── qwen2vl_7b.py         # Qwen2-VL-7B wrapper
├── 📁 scripts/                # Utility scripts
│   └── run_autocaption.sh    # Main execution script
├── 🐍 main.py                 # Main entry point
├── 🐍 mcts.py                 # MCTS algorithm implementation
├── 🐍 util.py                 # Utility functions
├── 📋 requirements.txt        # Python dependencies
├── ⚙️ setup.py                # Package setup
└── 📄 README.md               # This file
```

## 🎯 MCTS Action Types

AutoCaption uses 6 different action types for comprehensive video analysis:

1. **ACTION1**: Overall video description
2. **ACTION2**: Detail-focused observation (weighted selection)
3. **ACTION3**: Temporal perspective analysis
4. **ACTION4**: Spatial perspective analysis  
5. **ACTION5**: Background description
6. **ACTION6**: Camera movement analysis


## 📌 Citation

If you use AutoCaption or MCTS-VCB in your research, please cite our paper:

```bibtex
@misc{yu2025evaluatingmultimodallargelanguage,
    title={Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search}, 
    author={Linhao Yu and Xinguang Ji and Yahui Liu and Fanheng Kong and Chenxi Sun and Jingyuan Zhang and Hongzhi Zhang and V. W. and Fuzheng Zhang and Deyi Xiong},
    year={2025},
    eprint={2506.11155},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2506.11155},
}
```
