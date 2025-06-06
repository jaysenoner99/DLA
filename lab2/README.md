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

## 📂 Folder Structure

```bash
lab2/
│
├── dataloaders.py          # Utilities to load datasets
├── model.py                # CNN and Autoencoder implementations
├── fgsm.py                 # Perform targeted/untargeted attacks on pretrained classification networks
├── main.py                 # Main training entry point
├── train_and_test.py       # Training and testing routines
└── README.md               # This file
```
---

## 🚀 Usage

### 🔧 Example: Train a CNN with FGSM augmented samples

```bash
python main.py --model cnn --fgsm --epsilon 0.05 --use-wandb
```

### 🧪 Example: Perform a targeted attack on a pretrained CNN

```bash
python fgsm.py --model cnn --path path/to/weights --step 0.025 --max 0.1 --target-class 2 --use-wandb 
```

### 📊 Example: Evaluate OOD detection with max-softmax scoring 

```bash
python eval.py --model cnn --score softmax --path path_to_weights.pth
```

---

## 📥 Supported Arguments

### `main.py`

This script trains either a CNN classifier or an Autoencoder on CIFAR-10 with options for:

- Customizable learning rate, epochs, batch size, and learning rate schedules (step or cosine).
- Using FGSM adversarial samples during training (`--fgsm` flag).
- Device selection (CPU/GPU auto-detection).
- Reproducibility through random seed setting.
- Optional integration with Weights & Biases (`wandb`) for experiment tracking.
- Saving trained model weights automatically.
- Reporting test accuracy (top-1 and top-5 for CNN) or reconstruction metrics (for AE).

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

This script performs and evaluates Fast Gradient Sign Method (FGSM) adversarial attacks on pretrained CNN or Autoencoder models trained on CIFAR-10.

- Supports **targeted** and **untargeted** FGSM attacks.
- Generates adversarial examples by perturbing inputs in the direction of the gradient.
- Evaluates model accuracy under different perturbation strengths (`epsilon` values).
- For CNNs:
  - Measures classification accuracy and targeted attack success rate.
- For Autoencoders:
  - Measures reconstruction error (MSE) on perturbed inputs.
- Visualizes results:
  - Accuracy or loss vs epsilon plots.
  - Sample adversarial examples.
  - Targeted attack success rate
- Integrates with Weights & Biases (`wandb`) for logging.

Use this script to assess model robustness to adversarial perturbations and analyze the effect of varying attack strengths.

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

This script evaluates a pretrained model's ability to distinguish in-distribution (ID) from out-of-distribution (OOD) samples using simple scoring functions.

- Loads a CNN or autoencoder trained on CIFAR-10.
- Loads a set of fake (OOD) samples.
- Computes OOD scores using either:
  - **Max logit**
  - **Max softmax** 
- Evaluates the model’s performance in separating ID and OOD data.
- Supports Weights & Biases (`wandb`) for logging metrics and visualizations.

Use this script to test the robustness of your trained model against OOD inputs by analyzing score distributions and computing metrics like AUROC.

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

### 🔒 Training with Adversarial Examples

Results of training a Deep Convolutional Neural Network with (pink curve) and without(violet curve) FGSM augmented samples.
Note that for every batch processed by the model during training, there is a 50% probability of actually using FGSM samples. 
In the case of this experiments, the FGSM samples are generated with an epsilon value of 0.07.
This choice was made to ensure that the classification capabilities of the network are not entirely lost (which happens if we train the CNN with only FGSM samples)

<p align="center">
  <img src="images/train_loss_fgsm.png" width="33%" alt="Train loss">
  <img src="images/val_loss_fgsm.png" width="33%" alt="Validation Loss">
  <img src="images/val_acc_fgsm.png" width="33%" alt="Validation accuracy">
</p>

Observations:

- FGSM augmented training helps preventing overfitting, as it is observed by looking at the validation loss curve
- FGSM augmented training achieves an higher accuracy on both the test and the validation set.

| Model | FGSM Training | Epsilon | Test Accuracy | 
|-------|---------------|---------|----------------|
| CNN   | ❌             | -       | 64.78%          | 
| CNN   | ✅             | 0.07    | 67.93%          |

### 🚫 OOD Detection
Results of evaluating the OOD capabilities of different pretrained models (using the `eval.py` script). \
For these experiments, I used real data from CIFAR10 and OOD samples from [FakeData](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.FakeData.html).\
Here we report the results for a shallow CNN, a deep CNN, the same deep CNN trained with FGSM samples and an Autoencoder.

##### -Shallow CNN
<p align="center">
  <img src="images/cnn_evaluate_softmax.png" width="45%" alt="cnn OOD evaluation">
  <img src="images/ROC_cnn.png" width="45%" alt="cnn OOD evaluation">
</p>

##### -Deep CNN
<p align="center">
  <img src="images/deep_cnn_evaluate_softmax.png" width="45%" alt="cnn OOD evaluation">
  <img src="images/deep_cnn_ROC.png" width="45%" alt="cnn OOD evaluation">
</p>

##### -Deep CNN trained with FGSM samples 
<p align="center">
  <img src="images/deep_cnn_fgsm_softmax.png" width="45%" alt="cnn OOD evaluation">
  <img src="images/deep_cnn_fgsm_ROC.png" width="45%" alt="cnn OOD evaluation">
</p>

##### -Autoencoder
<p align="center">
  <img src="images/ae_evaluate.png" width="45%" alt="ae OOD evaluation">
  <img src="images/ROC_ae.png" width="45%" alt="ae OOD evaluation">
</p>

Observations:

- The histogram shows that both the shallow CNN and the deep CNN are not able to detect OOD samples.
- There is an improvement in OOD detection for the deep CNN model when the model is pretrained on FGSM samples. This observation can be further consolidated by comparing the AUROC scores of both models.
- The Autoencoder model has the best OOD detection performances out of all the tested models, proving that the Autoencoder model is more suited for the task of anomaly detection.

---

### 🖼️ Targeted attacks: FGSM augmented samples and model classifications

Results of attacking pretrained deep CNNs with a targeted FGSM attack with `target_class = 2`. \
Left : Pretrained deep CNN without augmentations.\
Right: Pretrained deep CNN with FGSM augmented samples (`epsilon = 0.07`).

<p align="center">
  <img src="images/cnn_targeted_samples.png" width="45%" alt="adv examples">
  <img src="images/cnn_fgsm_targeted_samples.png" width="45%" alt="adv examples">
</p>


### 📈 Targeted attacks: Attack success rate

<p align="center">
  <img src="images/attack_cnn_success_rate.png" width="45%" alt="adv examples">
  <img src="images/attack_cnn_fgsm_success_rate.png" width="45%" alt="adv examples">
</p>

Observations:

- Performing a targeted FGSM attack on the vanilla deep CNN results in an attack success rate that is steadly increasing with the strength of the perturbation `epsilon`.
- The attack success rate associated with the deep CNN trained using FGSM samples drops when approaching the value of `0.07`, which indicates that the model is more robust to samples that are perturbated with a value of `epsilon` that is similar to the one used during training.


🔗 More results available at [Wandb project - lab2](https://wandb.ai/jaysenoner/lab_2_DLA/workspace?nw=nwuserjaysenoner1999)

---

## 📚 References

- [Goodfellow et al. (2015)](https://arxiv.org/abs/1412.6572): Explaining and Harnessing Adversarial Examples  
- [Hendrycks & Gimpel (2017)](https://arxiv.org/abs/1610.02136): A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks

---

## 🧑‍💻 Author

Jay Senoner — [@jaysenoner99](https://github.com/jaysenoner99)
