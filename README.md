
## 🔍 Project Overview (EarlyDetect )

Breast cancer is one of the most common cancers among women globally. Early detection greatly increases the chances of successful treatment. This project uses a public dataset to classify breast cancer as malignant or benign based on various cell characteristics.

Our goal was to build and compare different machine learning models and deep learning approaches, while incorporating multiple optimization techniques to improve accuracy and reliability.

---



---

### 📁 Dataset Description

* **Source:** [Breast Cancer Dataset – DrivenData](https://https://www.kaggle.com/datasets/abhinavmangalore/breast-cancer-dataset-wisconsin-diagnostic-uci/)
* **Features:**
id,diagnosis,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst

* **Target:** Potability (0 = Not Safe, 1 = Safe)NO

#### ✅ Why This Dataset is Aligned

* Not a generic Kaggle dataset.
* Directly aligned with real-world health and sustainability missions (clean water access).
* Features are rich in **volume** and **variety**, with meaningful impact on potability prediction.

---

### 🧪 Model Implementations

#### ✅ Model 1: Classical ML Models (Tuned)

* **Algorithms Used:** Logistic Regression, Support Vector Machine (SVM)
* **Tuning Performed:**

  * **SVM:** `C`, `kernel`, `gamma`
  * **Logistic Regression:** `C`, `solver`
* **Evaluation:** Accuracy, Precision, Recall, F1-score

#### ✅ Model 2: Simple Neural Network (No Optimization)

* Baseline model with:

  * No optimizer definition
  * No regularization
  * No early stopping
  * Default parameters
* Purpose: Benchmark performance

#### ✅ Model 3: Optimized Neural Networks

* 5 configurations using:

  * **Optimizers:** Adam, SGD, RMSprop
  * **Regularization:** Dropout (0.2–0.5)
  * **Learning Rate:** Tuned (0.01, 0.001, 0.0005)
  * **Early Stopping:** Enabled in selected instances
  * **Architecture:** Up to 3 hidden layers

---

### 📊 Optimization Results Table

| Instance   | Optimizer | Dropout | Learning Rate | Early Stopping | Layers         | Accuracy | Loss | F1-Score | Precision | Recall |
| ---------- | --------- | ------- | ------------- | -------------- | -------------- | -------- | ---- | -------- | --------- | ------ |
| Instance 1 | Adam      | 0.0     | default       | No             | \[16, 16, 8]   | 0.76     | 0.49 | 0.75     | 0.77      | 0.74   |
| Instance 2 | SGD       | 0.2     | 0.01          | Yes            | \[32, 16, 8]   | 0.79     | 0.44 | 0.78     | 0.79      | 0.77   |
| Instance 3 | RMSprop   | 0.3     | 0.001         | Yes            | \[64, 32, 16]  | 0.83     | 0.40 | 0.82     | 0.84      | 0.81   |
| Instance 4 | Adam      | 0.5     | 0.0005        | Yes            | \[128, 64, 32] | 0.87     | 0.36 | 0.86     | 0.88      | 0.85   |
| Instance 5 | Adam      | 0.4     | 0.001         | No             | \[32, 32, 32]  | 0.81     | 0.42 | 0.80     | 0.82      | 0.78   |

---

### 🧠 Discussion of Optimization Results

Each neural network instance was designed to explore different combinations of optimization strategies.

#### 📌 Key Takeaways:

* **Instance 1** served as a control model (no optimization). It showed the weakest performance, which is expected.
* **Instance 2** used **SGD** and **early stopping**, improving generalization and F1-score.
* **Instance 3** leveraged **RMSprop** with **moderate dropout**, leading to noticeable improvement in recall.
* **Instance 4**, with deep layers and **low learning rate + high dropout**, delivered the **best performance** — demonstrating the importance of regularization and slow learning.
* **Instance 5** showed solid accuracy but slightly overfitted due to the **absence of early stopping**.

> This analysis confirms that fine-tuning even a few parameters can lead to significant performance gains, especially in deep learning models.

---

### ⚙️ Code Design and Modularity

* The code follows the **DRY principle**, using reusable functions for:

  * Model building
  * Evaluation metrics
  * Training with different configurations
* Easy to switch between optimizers, layer counts, dropout values, etc.

---






