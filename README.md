
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



| Training Instance | Optimizer | Regularizer | Epochs | Early Stopping | # Layers | Learning Rate | Accuracy | F1 Score | Recall  | Precision |
|-------------------|-----------|-------------|--------|----------------|----------|----------------|----------|----------|---------|-----------|
| Instance 1        | Default   | None        | 5      | No             | 1        | Default         | 0.7895   | 0.8286   | 0.8056  | 0.8529    |
| Instance 2        | Adam      | L2          | 30     | Yes            | 2        | 0.001           | 0.8772   | 0.9041   | 0.9167  | 0.8919    |
| Instance 3        | RMSprop   | L1          | 30     | Yes            | 2        | 0.0005          | 0.9035   | 0.9241   | 0.9306  | 0.9178    |
| Instance 4        |SGD      | l1       | 50     | Yes            | 3        | 0.0001          | 0.8772   | 0.9000   | 0.8750  | 0.9265    |





---

###  Discussion of Optimization Results


Best Performing Configuration

Instance 3 was the best performer, using RMSprop + L1 regularization, with the highest accuracy (90.35%) and F1 Score (92.41%).

The use of L1 regularization improved weight sparsity, while RMSprop helped adapt learning rates during training, stabilizing performance across epochs.

Optimizer and Regularization Impact

Instance 1 (Default setup) showed underfitting due to no optimizer configuration or regularization. It had the lowest performance overall.

Instance 2 with Adam + L2 significantly improved results by reducing overfitting through regularization and early stopping.

Instance 4 used SGD without regularization and performed decently, but lower recall suggests it struggled to detect some positives, likely due to slower convergence despite deeper layers.



---

<pre> ### How to Use This Repository Follow these steps to set up, run, and experiment with the models in this project. 
 
   1. Clone the Repository ```bash git clone https://https://github.com/nellyiya/EarlyDetect-.git cd EarlyDetect ``` 
   2. Install Dependencies Make sure you have Python 3.8+ installed, then install the required libraries: ```bash pip install -r requirements.txt ``` Or manually install: ```bash pip install numpy pandas scikit-learn matplotlib seaborn tensorflow ``` 
 3. Project Structure ``` ├── Notebook.ipynb # Main Jupyter notebook with training and evaluation ├── saved_models/ # Trained neural network models (.h5 files) ├── data/ # Dataset files (if applicable) ├── requirements.txt # Required Python packages └── README.md # Documentation ```
 4. Run the Notebook Open the notebook and run all cells: ```bash jupyter notebook Notebook.ipynb ``` You can view the training process, compare different neural network instances, and see evaluation results including accuracy, precision, recall, and F1-score.
5. Customize / Experiment - Change model architectures (number of layers, regularization) - Tune hyperparameters (learning rate, optimizer, epochs) - Add new metrics or visualizations 
 6. Results Summary Results are printed at the end of the notebook and summarized in the README under the "Training Instances" and "Error Analysis" sections. </pre>
---







