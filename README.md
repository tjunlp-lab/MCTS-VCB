# 🏷️ AutoCaption

📄 **[Paper: Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search
](https://arxiv.org/pdf/2506.11155)**  
🤗 **[Data](https://huggingface.co/datasets/HasuerYu/AutoCaption)**  

## 📌 Overview

MCTS-VCB is a fine-grained video captioning benchmark automatically constructed using a novel framework called AutoCaption. AutoCaption employs Monte Carlo Tree Search (MCTS) to generate rich, diverse, and detailed key point captions that thoroughly describe video content.

This benchmark enables comprehensive evaluation of Multimodal Large Language Models (MLLMs) on video understanding tasks.

## 🚀 Highlights

AutoCaption Framework: Iteratively constructs high-quality video descriptions using MCTS, covering objects, actions, environments, and more.

MCTS-VCB Benchmark: Contains diverse, multi-faceted video captions for robust MLLM evaluation.

Evaluation: Benchmarked over 20 MLLMs. Gemini-1.5-Pro achieved the top F1 score of 71.2.

Fine-tuning Results: InternVL2.5-8B fine-tuned on AutoCaption data achieved:

+25.0% improvement on MCTS-VCB

+16.3% improvement on DREAM-1K

## 📂 Repository Contents

`autocaption/`: Code for AutoCaption framework

`benchmark/`: Scripts and data for MCTS-VCB benchmark

`evaluation/`: Evaluation protocols and metrics

`models/`: Preprocessing and finetuning scripts for MLLMs


