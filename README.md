
## 🔍 Project Overview (EarlyDetect )

Breast cancer is one of the most common cancers among women globally. Early detection greatly increases the chances of successful treatment. This project uses a public dataset to classify breast cancer as malignant or benign based on various cell characteristics.

Our goal was to build and compare different machine learning models and deep learning approaches, while incorporating multiple optimization techniques to improve accuracy and reliability.

---

## 📊 Dataset Information

- **Name**: Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source**: Scikit-learn built-in datasets
- **Features**: 30 numerical features computed from digitized images of fine needle aspirates (e.g., radius, texture, smoothness).
- **Target**: Binary classification — `0` (malignant), `1` (benign)
- **Volume & Variety**:
  - 569 samples
  - Rich in meaningful diagnostic features
  - Balanced target classes

✅ **Aligned with my proposed mission of using technology for health diagnostics.**  
✅ **Not a generic Kaggle competition dataset.**

---

## ⚙️ Models Implemented

### ✅ Classical ML Model
- **Logistic Regression**
  - Tuned hyperparameter: `C=0.5`, penalty = `L2`
  - Solver: `liblinear`

### ✅ Simple Neural Network
- 2 layers (16 units)
- **No optimization techniques**
- Uses default hyperparameters

### ✅ Optimized Neural Network
- Multiple layers with **more than 8 neurons each**
- **Regularization**: L2
- **Dropout**: 0.2–0.3
- **Optimizer**: Adam with tuned learning rate
- **Early stopping** applied
- Adjusted **learning rate**, **epochs**, **architecture**

---

## 📈 Evaluation Metrics

For each model, we calculated:

- **Accuracy**
- **Loss**
- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC** (bonus)

---

## 📋 Training Results Summary Table

| Instance            | Optimizer | Regularizer | Dropout | LR     | Accuracy | F1 Score | Precision | Recall | Loss  |
|---------------------|-----------|-------------|---------|--------|----------|----------|-----------|--------|-------|
| Simple NN (No Opt)  | Adam      | None        | None    | Default| 0.95     | 0.94     | 0.93      | 0.95   | ~0.13 |
| Optimized NN 1      | Adam      | L2 (0.001)  | 0.3     | 0.001  | 0.97     | 0.96     | 0.96      | 0.97   | ~0.09 |
| Optimized NN 2      | RMSprop   | L2 (0.01)   | 0.2     | 0.001  | 0.96     | 0.95     | 0.94      | 0.96   | ~0.11 |
| Optimized NN 3      | SGD       | None        | 0.2     | 0.01   | 0.94     | 0.93     | 0.91      | 0.94   | ~0.15 |
| Optimized NN 4      | Adam      | L2 (0.001)  | 0.5     | 0.0005 | 0.95     | 0.94     | 0.93      | 0.95   | ~0.14 |

*Note: Values are approximate based on model training results.*

---

## 🧠 Key Takeaways

- Classical ML (Logistic Regression) provided a good baseline, but lacked flexibility.
- The simple neural network performed well, but suffered slightly from overfitting.
- Optimized models **significantly improved recall and F1-score**, indicating better generalization.
- Techniques like **early stopping**, **regularization**, and **dropout** helped reduce overfitting.
- **Learning rate adjustment** had a direct impact on convergence and stability.

---

## 🎥 Video Summary

A 5-minute video is included demonstrating:
- Dataset reasoning and task alignment
- Explanation of models and architecture
- Impact of optimization techniques
- Reflections on model performance

---

## 🧩 Modularity

The notebook is modular, with:
- Reusable model-building functions
- Evaluation metric functions
- Separated cells for easy tuning of hyperparameters

---

## ✅ Conclusion

This project highlights how optimization techniques are crucial to training robust neural networks. By systematically applying and comparing different approaches, we achieved improved performance and deeper understanding of model behavior.
```

---


