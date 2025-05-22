# Lab 2: Adversarial Machine Learning, OOD Detection, and Adversarial Robustness

This folder contains the implementation of the second laboratory assignment for the **Deep Learning Applications** course (University of Florence, MSc in Artificial Intelligence). The focus of this lab is on understanding the vulnerabilities of deep learning models to out-of-distribution (OOD) inputs and adversarial attacks, as well as exploring methods to enhance model robustness.

---

## 📦 Contents

- **Model Training**:
  - Train either a Convolutional Neural Network (CNN) or an Autoencoder (AE) on MNIST, CIFAR10, or CIFAR100.
  - Optional adversarial training using Fast Gradient Sign Method (FGSM).
- **Adversarial Attacks**:
  - Targeted and untargeted attacks with different epsilon values.
- **OOD Detection**:
  - Max logit and max softmax scoring to distinguish in-distribution vs OOD samples.
- **Evaluation**:
  - Evaluate the performance of models on adversarial and OOD samples.

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

Clone the repository:

```bash
git clone https://github.com/jaysenoner99/DLA.git
cd DLA/lab2
```

---

## 🚀 Usage

### 🔧 Training

```bash
python main.py --model cnn --fgsm --epsilon 0.05 --use-wandb
```

### 🧪 FGSM Attacks

```bash
python fgsm.py --model cnn --step 0.01 --max 0.2 --target-class 3 --use-wandb
```

### 📊 Evaluation (OOD Detection)

```bash
python eval.py --model cnn --score softmax --path path_to_weights.pth
```

---

## 📥 Supported Arguments

### `main.py`

| Argument       | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `--lr`         | Initial learning rate (default: `0.0001`)                                  |
| `--epochs`     | Total number of training epochs (default: `50`)                            |
| `--schedule`   | Epochs to reduce learning rate by 10x (default: `[125, 175]`)              |
| `--batch-size` | Batch size for training (default: `256`)                                   |
| `--cos`        | Use cosine annealing learning rate schedule                                |
| `--device`     | Device to use (`cpu` or `cuda`, auto-selected)                             |
| `--val-split`  | Proportion of training data for validation (default: `0.1`)                |
| `--seed`       | Random seed for reproducibility (default: `69`)                            |
| `--use-wandb`  | Enable Weights & Biases logging                                            |
| `--model`      | Choose `cnn` or `ae`                                                       |
| `--fgsm`       | Enable adversarial training using FGSM                                     |
| `--epsilon`    | Epsilon for FGSM perturbation (default: `0.07`)                            |
| `--path`       | Path to model weights to load                                              |

---

### `fgsm.py`

| Argument         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `--device`       | Device to use (`cpu` or `cuda`, auto-selected)                             |
| `--batch-size`   | Batch size (default: `1`)                                                   |
| `--val-split`    | Proportion of training data for validation (default: `0.1`)                |
| `--seed`         | Random seed (default: `69`)                                                 |
| `--use-wandb`    | Enable Weights & Biases logging                                            |
| `--model`        | Choose `cnn` or `ae`                                                       |
| `--step`         | Step increment for epsilon values (default: `0.025`)                       |
| `--max`          | Maximum epsilon value for perturbation (default: `0.2`)                    |
| `--path`         | Path to model weights to load                                              |
| `--target-class` | Class index for targeted FGSM attacks                                      |

---

### `eval.py`

| Argument       | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `--device`     | Device to use (`cpu` or `cuda`, auto-selected)                             |
| `--batch-size` | Batch size (default: `256`)                                                 |
| `--seed`       | Random seed (default: `69`)                                                 |
| `--epochs`     | Number of evaluation iterations (default: `200`)                           |
| `--val-split`  | Validation data proportion (default: `0.1`)                                 |
| `--use-wandb`  | Enable Weights & Biases logging                                            |
| `--model`      | Choose `cnn` or `ae`                                                       |
| `--score`      | Scoring function: `logit` or `softmax`                                     |
| `--path`       | Path to model weights to load                                              |

---

## 📊 Results Summary

### 🔒 Adversarial Robustness

| Model | FGSM Training | Epsilon | Test Accuracy | Adversarial Accuracy |
|-------|---------------|---------|----------------|----------------------|
| CNN   | ❌             | -       | 91.2%          | 12.7%                |
| CNN   | ✅             | 0.07    | 89.5%          | 74.3%                |

### 🚫 OOD Detection

| Model | Scoring Function | OOD Dataset    | AUROC |
|-------|------------------|----------------|--------|
| CNN   | Softmax          | FashionMNIST   | 92.5%  |
| AE    | Logit            | SVHN           | 88.1%  |

---

## 🖼️ Example Results

<p align="center">
  <img src="assets/adversarial_example.png" width="45%" alt="Adversarial Attack Example">
  <img src="assets/ood_detection.png" width="45%" alt="OOD Detection ROC Curve">
</p>

---

## 📚 References

- [Goodfellow et al. (2015)](https://arxiv.org/abs/1412.6572): Explaining and Harnessing Adversarial Examples  
- [Hendrycks & Gimpel (2017)](https://arxiv.org/abs/1610.02136): A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks

---

## 🧑‍💻 Author

Jaysen Oner — [@jaysenoner99](https://github.com/jaysenoner99)
