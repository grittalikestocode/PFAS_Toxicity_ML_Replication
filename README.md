# PFAS_Toxicity_ML_Replication

This repository contains a replication and extension of the study by Feinstein et al. (2021), "Uncertainty-Informed Deep Transfer Learning of Perfluoroalkyl and Polyfluoroalkyl Substance Toxicity", originally published in Journal of Chemical Information and Modeling.

Our work reproduces the core models and analyses from the paper, while addressing reproducibility issues and providing further insights into dataset bias, model calibration, and prediction consistency.

## 📌 Brief

This project was completed as part of a course in the Master’s in Applied Data Science program at the University of Gothenburg, Sweden.

The objective was to:

1. Replicate the original AI4PFAS deep learning workflow.

2. Reproduce key results using GCN, GP, RF, and DNN models.

3. Analyze uncertainty, bias, and reproducibility challenges.

4. Explore performance through visualizations, loss distributions, and chemical similarity.

## 📚 Original Study Citation

Feinstein, J., Sivaraman, G., Picel, K., Peters, B., Vázquez-Mayagoitia, Á., Ramanathan, A., MacDonell, M., Foster, I., & Yan, E. (2021).
Uncertainty-Informed Deep Transfer Learning of Perfluoroalkyl and Polyfluoroalkyl Substance Toxicity.
Journal of Chemical Information and Modeling.
https://doi.org/10.1021/acs.jcim.1c01204

## 📂 Repository Structure
~~~
ai4pfas/
├── data/                    # Processed and raw datasets
│   ├── benchmark-models/
│   ├── deep-ensemble/
│   ├── latent-space/
│   ├── preprocessed/
│   ├── selective-net/
│   └── transfer-learning/
├── notebooks/              # Jupyter notebooks for all models
│   ├── dnn/
│   ├── gcn/
│   ├── gp/
│   ├── rf/
├── src/                    # Core Python scripts and model definitions
├── media/                  # Visuals and plots
├── environment.yml         # Conda environment setup
├── LICENSE
└── README.md
~~~
## 📊 Dataset

The LDToxDB dataset was used, created by merging LD₅₀ data from:

EPA Toxicity Estimation Software Tool (TEST)

NIH Collaborative Acute Toxicity Modeling Suite (CATMoS)

National Toxicology Program (NTP)

Data was filtered by removing duplicates (using InChIKey), and converted to LD₅₀ in -log(mol/kg) scale. Mordred descriptors were used for molecular feature extraction.

## ⚙️ Installation

1. Install Anaconda

2. Create and activate the conda environment:
~~~
conda create -n ai4pfas -f environment.yml
conda activate ai4pfas
~~~

## ✅ Our Contributions

Reproduced model training pipelines for GCN, GP, RF, and DNN using the original datasets.

Conducted additional analysis on:

Dataset bias and chemical similarity (Tanimoto scores).

Calibration and uncertainty metrics.

Confusion matrices, loss distributions, and toxicity rank correlations.

Resolved environment and reproducibility issues from the original codebase.

Improved data preprocessing and organization for clarity.

## 🤝 Acknowledgments

We thank the original authors for open-sourcing their code and enabling further exploration of this critical topic in environmental toxicology.

