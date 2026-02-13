# RLKGC

<p align="center">
<b>RLKGC: Reinforcement Learning Retrieval with Large Language Models for Knowledge Graph Completion</b><br>
Accepted at <b>PAKDD 2026</b>
</p>

---

## 📌 Overview

RLKGC is a unified framework that integrates:

- 🔵 **multihopRL** — Reinforcement Learning with Reward Shaping for multi-hop knowledge graph reasoning  
- 🟢 **KGC** — LLM-based reasoning with type-aware subgraph and path-aware retrieval  

The complete pipeline consists of two stages:

1. **RL-based Path Retrieval**
2. **LLM-based Reasoning over Retrieved Paths**

---

## 📁 Repository Structure

```
RLKGC/
├── multihopRL/      # RL + reward shaping module
├── KGC/             # LLM-based reasoning module
└── README.md
```

---

# ⚙️ Environment Setup

Recommended configuration:

- Python 3.10  
- PyTorch ≥ 2.2  
- CUDA-enabled GPU  

Create environment:

```
conda create -n rlkgc python=3.10
conda activate rlkgc
pip install -r requirements.txt
pip install sentence_transformers
```

---

# 📂 Datasets

We evaluate RLKGC on:

- FB15k-237  
- WN18RR  
- NELL-995  

⚠️ Datasets are **NOT included** in this repository.

Please download `data-release.tgz` from the official MultiHopKG repository  
[Download here](PASTE_MULTIHOPKG_DATA_LINK)

Unpack:

```
tar xvzf data-release.tgz
```

Place the extracted datasets under:

```
multihopRL/data/
```

For the LLM module, download the full dataset and instruction files from  
[Dataset & Instructions link](PASTE_KGC_DATA_LINK)

Then copy the `datasets/` and `instructions/` folders into:

```
KGC/
```

---

# 🔵 Stage 1: multihopRL (RL-based Retrieval)

## 1️⃣ Process Data

```
./experiment.sh configs/<dataset>.sh --process_data 0
```

Available datasets:

- fb15k-237  
- wn18rr  
- nell-995  

---

## 2️⃣ Train Embedding Model

Supported embedding models:

- distmult  
- complex  
- conve  

```
./experiment-emb.sh configs/<dataset>-<embedding_model>.sh --train <gpu-ID>
```

Optional evaluation:

```
./experiment-emb.sh configs/<dataset>-<embedding_model>.sh --inference 0
```

---

## 3️⃣ Train RL + Reward Shaping

Edit configuration file:

```
nano configs/<dataset>-rs.sh
```

Set embedding checkpoint path:

```
complex_state_dict_path="model/<your_pretrained_embedding>/best_dev_iteration.dat"
```

Train:

```
./experiment-rs.sh configs/<dataset>-rs.sh --train 0
```

Inference:

```
./experiment-rs.sh configs/<dataset>-rs.sh --inference 0
```

Generated path file:

```
multihopRL/outputs/<dataset>_cats.jsonl
```

---

# 🟢 Stage 2: KGC (LLM-based Reasoning)

## 1️⃣ Install Dependencies

```
pip install -r requirements.txt
pip install sentence_transformers
```

---

## 2️⃣ Convert Retrieved Paths (Inductive Setting)

```
python tools/convert_fb15k237_paths.py
```

This generates:

```
datasets/<dataset>/paths/close_path.json
```

---

## 3️⃣ (Optional) Build Instructions

```
python build_instructions.py \
  --dataset FB15k-237-subset \
  --train_size full \
  --prompt_type CATS \
  --subgraph_type combine \
  --neg_num 12
```

---

# 🤖 LLM Setup

Experiments can be reproduced using:

- Qwen2-7B-Instruct  
- Meta-Llama-3-8B-Instruct  
- Meta-Llama-3-1.5B-Instruct  

Download model checkpoints from:

- Qwen2-7B-Instruct: [Model Link](PASTE_MODEL_LINK)
- Meta-Llama-3-8B-Instruct: [Model Link](PASTE_MODEL_LINK)
- Meta-Llama-3-1.5B-Instruct: [Model Link](PASTE_MODEL_LINK)

Update:

```
KGC/data_manager.py
```

Set:

```
LLM_PATH = "<your_local_model_path>"
```

---

# 🚀 Inference

Example (Inductive Setting):

```
python prediction.py \
  --dataset <your_dataset> \
  --setting inductive \
  --train_size full \
  --model_name <model_path> \
  --llm_type sft \
  --prompt_type CATS \
  --subgraph_type combine \
  --path_type degree
```

To run in transductive setting:

```
--setting transductive
```

---

# ⚠️ Important Note for NELL-995

For NELL-995, the original training data is split into `train.triples` and `dev.triples`.  
For final evaluation, the model must be trained using both files combined.

To obtain correct test results, include the `--test` flag in all preprocessing, training, and inference commands.

Process data:

```
./experiment.sh configs/nell-995.sh --process_data <gpu-ID> --test
```

Train embedding model:

```
./experiment-emb.sh configs/nell-995-conve.sh --train <gpu-ID> --test
```

Train RL + Reward Shaping:

```
./experiment-rs.sh configs/NELL-995-rs.sh --train <gpu-ID> --test
```

Inference:

```
./experiment-rs.sh configs/NELL-995-rs.sh --inference <gpu-ID> --test
```

⚠️ During development, leave out the `--test` flag.

---

# 📊 Outputs

RL retrieval outputs:

```
multihopRL/outputs/<dataset>_cats.jsonl
```

LLM predictions:

```
KGC/outputs/
```

---

# 📜 Citation

If you use this code, please cite:

```
@inproceedings{RLKGC2026,
  title={RLKGC: Reinforcement Learning Retrieval with Large Language Models for Knowledge Graph Completion},
  booktitle={PAKDD},
  year={2026}
}
```

---

# 🔒 Notes

- Datasets, logs, and model checkpoints are excluded from this repository.
- Ensure correct GPU assignment before training.
- Adjust training epochs based on development performance.
