# COMP8800 Educational Probing

## Dataset

This project uses the PISA 2018 dataset, which can be downloaded [here](https://webfs.oecd.org/pisa2022/index.html).

The project focuses on four countries: Australia, Brazil, Japan, and Switzerland. Country-specific subsets derived from the original PISA 2018 student questionnaire data are stored under `./data`.

## Reproducibility

The code is written in Python 3.x and was run on a university-managed GPU cluster `Cluster1` using Slurm.

### Main file

`./pisa-escs-main.py`

This file contains the main pipeline for prompt construction, activation extraction, probing, and the initial intervention generation step.

Recommended resources:
- NVIDIA A100, A6000, or L40S GPU
- At least 80GB GPU memory
- Around 100GB system memory
- Around 100GB disk space

Estimated total runtime: 40+ hours, depending on the model, country, and compute node.

### Label Transformations

`./pisa-escs-label-transformations.py`

This file contains the label-transformation robustness checks, including the original ESCS labels, randomly permuted labels, sine-transformed labels, and cubic-transformed labels.

### Intervention

`./pisa-escs-intervention.py`

This file contains the intervention scoring and analysis steps. Each step in this file is independent and should not be run all at once. Only one step should be uncommented and run at a time.

### Results
Figures are stored under `./figures`, and all intermediate results are stored under `./results`.
