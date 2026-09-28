# Heart Disease Prediction using MLP & PyTorch

An interactive machine learning web application built with **Streamlit** and **PyTorch** that predicts the presence of heart disease based on 13 patient clinical parameters. The model architecture is a Multi-Layer Perceptron (MLP) trained and hyperparameter-tuned using **Ray Tune**.

---

## Project Overview

Heart disease is one of the leading causes of mortality worldwide. Early detection can significantly improve patient outcomes. This project provides a user-friendly diagnostic web interface where medical practitioners or individuals can input clinical health indicators (such as age, cholesterol levels, resting blood pressure, max heart rate, etc.) and receive an instant AI-assisted risk assessment.

### Key Highlights
- **Deep Learning Model:** PyTorch Multi-Layer Perceptron (`MLP`) with ReLU activations and Sigmoid output.
- **Hyperparameter Tuning:** Optimized using **Ray Tune** for optimal learning rate, Adam betas, and weight initialization.
- **Data Preprocessing:** Standardized using `scikit-learn`'s `StandardScaler` to handle multi-scale clinical features.
- **Web Interface:** Interactive Streamlit web application with modern card components, input validation, and real-time risk scoring.

---
## Startup instructions
- Use the command ```git clone https://github.com/FatimaSaadat17/HeartDiseaseMLP.git``` in your terminal, ensure that you are in your working directory
  
  
- install all the required packages to run the model
```cd HeartDiseaseMLP```
```pip install -r requirements.txt```

- ensure streamlit is installed by using the command pip install streamlit --upgrade
```streamlit run app.py```


## Repository Structure

```text
├── heart.csv                # Dataset containing 1,025 medical records (13 features + 1 target)
├── HeartDiseaseNN.ipynb     # Jupyter Notebook for data exploration, model training, and Ray Tune tuning
├── model.py                 # PyTorch MLP class definition and prediction logic
├── app.py                   # Main Streamlit web application
├── model_params/            # Saved Ray Tune checkpoint directories and trained model weights (model.pth)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
