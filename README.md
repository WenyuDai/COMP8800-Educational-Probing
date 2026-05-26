# COMP8800 Educational Probing

## 1. Dataset

This project uses the PISA 2018 dataset, which can be downloaded [here](https://webfs.oecd.org/pisa2022/index.html).

The project focuses on four countries: Australia, Brazil, Japan, and Switzerland. Country-specific subsets derived from the original PISA 2018 student questionnaire data are stored under `./data`.

## 2. Reproducibility

The code is written in Python 3.x and was run on a university-managed GPU cluster `Cluster1` using Slurm.

### 2.1 Main file

`./pisa-escs-main.py`

This file contains the main pipeline for prompt construction, activation extraction, probing, and the initial intervention generation step.

Recommended resources:
- NVIDIA A100, A6000, or L40S GPU
- At least 80GB GPU memory
- Around 100GB system memory
- Around 100GB disk space

Estimated total runtime: 40+ hours, depending on the model, country, and compute node.

### 2.2 Label Transformations

`./pisa-escs-label-transformations.py`

This file contains the label-transformation robustness checks, including the original ESCS labels, randomly permuted labels, sine-transformed labels, and cubic-transformed labels.

### 2.3 Intervention

`./pisa-escs-intervention.py`

This file contains the intervention scoring and analysis steps. Each step in this file is independent and should not be run all at once. Only one step should be uncommented and run at a time.

## 3. Results
Figures are stored under `./figures`, and all intermediate results are stored under `./results`.

### 3.1 Notes on Intermediate Feature Files

The `results/*_features.pkl` files are not included in this repository because they are extremely large, often ranging from approximately 1 GB to nearly 3 GB per file. These files are generated from `./pisa-escs-main.py` and store intermediate attention-head activation representations extracted from the language models during the probing stage.

Each feature file contains `(features, labels)`, where `features.shape = (num_samples, 1, num_layers, num_heads, head_dim)`.

For the 7B models used in this project (`Llama-2-7B-Chat`, `Mistral-7B-Instruct-v0.1`, and `Vicuna-7B-v1.5`), this is typically `(num_students, 1, 32, 32, 128)`.

The dimensions correspond to:
- `num_students`: the number of PISA student samples for a given country
- `1`: only the final-token representation is retained
- `32`: transformer layers
- `32`: attention heads per layer
- `128`: hidden dimension for each attention head

The stored activations correspond to the attention-head outputs of the final token in each prompt. These intermediate neural representations are later used for linear probing, layer/head ranking, intervention vector construction, and activation steering experiments. They are excluded from version control because their file sizes exceed the practical limits for pushing to Git.

## 4. Acknowledgement

This project adapts the probing and intervention workflow from Kim, Evans, and Schein (2025), *Linear Representations of Political Perspective Emerge in Large Language Models* (ICLR 2025), for an educational PISA (ESCS) setting: https://openreview.net/forum?id=rwqShzb9li