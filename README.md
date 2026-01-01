# ML Implementations

This repository contains clean, modular implementations of classic machine learning algorithms using PyTorch and supporting libraries. The code is organized by algorithm type, with a focus on clarity, extensibility, and educational value.

## Features

- **Regression Algorithms**
  - Linear Regression (Least Squares, Ridge, Lasso)
  - Kernelized regression (polynomial features)
- **Classification Algorithms**
  - Logistic Regression (SGD, Adam, Convex Programming)
  - Decision_trees (Gini, train error, entropy gain measures)
- **Clustering Algorithms**
  - K-Means (Lloyd's algorithm, k-means++ initialization)
- **Dimensionality Reduction**
  - Principle Component Analysis 
- **Utilities**
  - Data loading and preprocessing
  - Evaluation metrics and results classes
  - Profiling and visualization tools

## Usage
1. uv sync
2. uv run -m src.regression.linear_regression --device cuda

## References
[Understanding Machine Learning: From Theory to Algorithms by Shalev-Shwartz and Ben-David](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)
[The Elements of Statistical Learning by Hastie,Tibshirani,Friedman] (https://hastie.su.domains/ElemStatLearn/)