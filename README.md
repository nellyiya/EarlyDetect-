
##  Project Overview (EarlyDetect )

Breast cancer diagnosis is challenging and requires accurate early detection to improve patient outcomes. This project develops a machine learning model to classify tumors as malignant or benign using clinical data. The dataset contains key features from breast cancer patients to train and evaluate the model. The goal is to support faster and more reliable diagnosis through automated prediction. This solution aims to assist healthcare professionals in making informed decisions efficiently and improving patient care.

---

###  Dataset Description

* **Source:** [Breast Cancer Dataset](https://www.kaggle.com/datasets/abhinavmangalore/breast-cancer-dataset-wisconsin-diagnostic-uci)
* **Features:**
id, diagnosis, radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave points_worst, symmetry_worst, fractal_dimension_worst
* **Target:** Diagnosis (M = Malignant, B = Benign)

####  Why This Dataset is Aligned

* Not a generic dataset; specific to breast cancer diagnosis.
* Directly aligned with real-world healthcare missions focusing on early cancer detection.
* Features are rich in clinical variety and volume, enabling robust model training and impactful medical insights.

---

###  Model Implementations

####  Model 1: Classical ML Models (Tuned)

* **Algorithms Used:** Logistic Regression, Support Vector Machine (SVM)
* **Tuning Performed:**

  * **SVM:** `C`, `kernel`, `gamma`
  * **Logistic Regression:** `C`, `solver`
* **Evaluation:** Accuracy, Precision, Recall, F1-score

####  Model 2: Simple Neural Network (No Optimization)

* Baseline model with:

  * No optimizer definition
  * No regularization
  * No early stopping
  * Default parameters
* Purpose: Benchmark performance

####  Model 3: Optimized Neural Networks

* 5 configurations using:

  * **Optimizers:** Adam, SGD, RMSprop
  * **Regularization:** Dropout (0.2–0.5)
  * **Learning Rate:** Tuned (0.01, 0.001, 0.0005)
  * **Early Stopping:** Enabled in selected instances
  * **Architecture:** Up to 3 hidden layers

---

###  Optimization Results Table
###  Training Instances Summary

### 🧠 Training Instances Summary

| Instance    | Optimizer | Regularizer | Epochs | Early Stopping | Layers | Learning Rate | Accuracy | F1 Score | Recall  | Precision |
|-------------|-----------|-------------|--------|----------------|--------|----------------|----------|----------|---------|-----------|
| Instance 1  | Adam      | None        | 5      | Yes            | 1      | 0.001          | 0.7368   | 0.8148   | 0.9167  | 0.7333    |
| Instance 2  | Adam      | L2          | 30     | Yes            | 2      | 0.001          | 0.8684   | 0.8966   | 0.9028  | 0.8904    |
| Instance 3  | RMSprop   | L1          | 30     | Yes            | 2      | 0.0005         | 0.9035   | 0.9272   | 0.9722  | 0.8861    |
| Instance 4  | Adam      | None        | 50     | Yes            | 3      | 0.0001         | 0.8421   | 0.8875   | 0.9861  | 0.8068    |






---

###  Discussion of Optimization Results
---

<pre>
### Error Analysis and Evaluation of Optimization Strategies

This section analyzes the performance of each neural network training instance using core classification metrics: Accuracy, F1 Score, Recall, Precision, and Loss (inferred through performance behavior and regularization effects).

---

### Summary of Training Instances

#### Instance 1 – Baseline Model  
- Setup: 1 hidden layer, trained for 5 epochs using Adam, with no regularization, and early stopping  
- Learning Rate: 0.001  
- Performance:  
  - Accuracy: 0.7368  
  - F1 Score: 0.8148  
  - Recall: 0.9167  
  - Precision: 0.7333  
- Interpretation: The model underfits due to low complexity and short training time. It catches most positive cases (high recall), but misclassifies many negatives (low precision).  

---

#### Instance 2 – Regularized Adam  
- Setup: 2 layers, trained for 30 epochs with L2 regularization, Adam optimizer, and early stopping  
- Learning Rate: 0.001  
- Performance:  
  - Accuracy: 0.8684  
  - F1 Score: 0.8966  
  - Recall: 0.9028  
  - Precision: 0.8904  
- Interpretation: Balanced and strong performance. L2 regularization and early stopping reduced overfitting and improved generalization.  

---

#### Instance 3 – Best Model (RMSprop + L1)  
- Setup: 2 layers, trained for 30 epochs using RMSprop, L1 regularization, and early stopping  
- Learning Rate: 0.0005  
- Performance:  
  - Accuracy: 0.9035  
  - F1 Score: 0.9272  
  - Recall: 0.9722  
  - Precision: 0.8861  
- Interpretation: This is the top-performing model. Sparse weights from L1 and adaptive learning from RMSprop produced high accuracy and recall with stable training.  

---

#### Instance 4 – Deep Model Without Regularization  
- Setup: 3 layers, trained for 50 epochs using Adam, with no regularization, early stopping, and a very small learning rate  
- Learning Rate: 0.0001  
- Performance:  
  - Accuracy: 0.8421  
  - F1 Score: 0.8875  
  - Recall: 0.9861  
  - Precision: 0.8068  
- Interpretation: The deeper network and long training improved recall greatly but hurt precision due to overfitting. Lack of regularization caused more false positives.  

---

<pre> ### How to Use This Repository Follow these steps to set up, run, and experiment with the models in this project. 
 
   1. Clone the Repository ```bash git clone https://https://github.com/nellyiya/EarlyDetect-.git cd EarlyDetect ``` 
   2. Install Dependencies Make sure you have Python 3.8+ installed, then install the required libraries: ```bash pip install -r requirements.txt ``` Or manually install: ```bash pip install numpy pandas scikit-learn matplotlib seaborn tensorflow ``` 
 3. Project Structure ``` ├── Notebook.ipynb # Main Jupyter notebook with training and evaluation ├── saved_models/ # Trained neural network models (.h5 files) ├── data/ # Dataset files (if applicable) ├── requirements.txt # Required Python packages └── README.md # Documentation ```
 4. Run the Notebook Open the notebook and run all cells: ```bash jupyter notebook Notebook.ipynb ``` You can view the training process, compare different neural network instances, and see evaluation results including accuracy, precision, recall, and F1-score.
5. Customize / Experiment - Change model architectures (number of layers, regularization) - Tune hyperparameters (learning rate, optimizer, epochs) - Add new metrics or visualizations 
 6. Results Summary Results are printed at the end of the notebook and summarized in the README under the "Training Instances" and "Error Analysis" sections. </pre>
---







