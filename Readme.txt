Wheat Leaf Pest and Disease Classification using Machine Learning

Project Overview:
This project implements multiple machine learning models to classify wheat leaf images into healthy and disease-affected categories. The goal is to identify pests and diseases accurately to support precision agriculture.

Models included:
- Support Vector Machine (SVM)
- XGBoost
- K-Nearest Neighbors (KNN)
- Random Forest
- Logistic Regression
- Gradient Boosting
- Decision Tree

Performance metrics computed:
Accuracy, Precision, Recall, F1 Score, Matthews Correlation Coefficient (MCC), AUC, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R2 Score.

Confusion matrices and comparison plots are also generated.

---------------------------------------------------------------
Dataset:
The Wheat Leaf Dataset used in this project is from Kaggle:

https://www.kaggle.com/datasets/olyadgetch/wheat-leaf-dataset

Dataset Description:
- Images of wheat leaves categorized by health and disease conditions.
- Diverse environmental and lighting conditions.
- Includes healthy leaves and multiple disease classes.

How to Download:
1. Visit the Kaggle link above.
2. Create/sign in to a Kaggle account.
3. Download the dataset ZIP file.
4. Extract the dataset locally.
5. Update the dataset path in the code accordingly.

---------------------------------------------------------------
Project Structure:
- data/                 : Contains extracted dataset folders (e.g., Healthy/, Disease1/, Disease2/)
- models/               : Optional folder to save trained models
- scripts/              : Python scripts for training and evaluation
  - train_models.py
  - evaluate_models.py
- requirements.txt      : Python package dependencies
- README.txt            : This file

---------------------------------------------------------------
Installation and Setup:
1. Clone or download this repository.
2. Create and activate a Python virtual environment (recommended).
3. Install dependencies:
   pip install -r requirements.txt
4. Download and extract the Wheat Leaf Dataset into the data/ folder.
5. Adjust dataset paths in the scripts if needed.

---------------------------------------------------------------
Usage:
1. Run training script to train all ML models:
   python scripts/train_models.py

2. Run evaluation and visualization script:
   python scripts/evaluate_models.py

3. Results and plots will display and/or save to disk.

---------------------------------------------------------------
Dependencies:
- Python 3.8+
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

---------------------------------------------------------------
Notes:
- Images may require resizing and preprocessing before model training.
- This project focuses on classical ML models; deep learning is optional.
- Cross-validation can be added for robustness.

---------------------------------------------------------------
Contact:
Your Name
your.email@example.com
GitHub: https://github.com/yourusername

---------------------------------------------------------------
