# Explainable AI for Diabetes Risk Prediction

This repository implements an explainable AI pipeline for predicting diabetes using multiple datasets. It includes data preprocessing, MLP classification, and explainability with SHAP, LIME, and DiCE.

## Structure
- `data/`: Input datasets
- `notebooks/`: Jupyter notebooks for end-to-end runs
- `src/`: Python modules (preprocessing, modeling, explainability)
- `results/`: Generated plots and figures
- `docs/`: Optional documentation and manuscript

## Getting Started
```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook DiabetesRiskPipeline.ipynb
```

## Dependencies
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- shap
- lime
- dice-ml
- matplotlib
- seaborn
