# Road Extraction from Satellite Imagery — A Step-by-Step Tutorial

This tutorial walks through building a **semantic segmentation** pipeline to automatically detect road networks in satellite images. By the end, you will understand how to load geospatial data, define a custom PyTorch dataset, train a U-Net model with TorchGeo and PyTorch Lightning, evaluate results, and visualize predictions.

**What you will build:** A model that takes a satellite image as input and outputs a binary mask where every pixel is classified as either *road* or *background*.

---

## Table of Contents

1. [Background](#1-background)
2. [The Dataset](#2-the-dataset)
3. [Setup](#3-setup)
4. [Step 1 — Loading the Data](#step-1--loading-the-data)
5. [Step 2 — The Lightning DataModule](#step-2--the-lightning-datamodule)
6. [Step 3 — Configuring the Model](#step-3--configuring-the-model)
7. [Step 4 — Training](#step-4--training)
8. [Step 5 — Evaluation](#step-5--evaluation)
9. [Step 6 — Visualizing Predictions](#step-6--visualizing-predictions)
10. [Results](#results)
11. [Citation](#citation)

---

## 1. Background

### What is Semantic Segmentation?

Unlike image classification (one label per image) or object detection (bounding boxes), **semantic segmentation** assigns a class label to *every single pixel*. For road extraction, the output is a mask the same size as the input image where:

- `1` = road pixel
- `0` = background pixel

### Why U-Net?

[U-Net](https://arxiv.org/abs/1505.04597) is the standard architecture for segmentation tasks. It has two parts:

- **Encoder** — progressively downsamples the image to extract features (what is in the image)
- **Decoder** — progressively upsamples back to the original resolution (where exactly)
- **Skip connections** — pass fine-grained spatial details from encoder layers directly to the matching decoder layers, preserving sharp edges like road boundaries

### Why MobileNetV2?

Instead of a standard encoder, we use a pretrained [MobileNetV2](https://arxiv.org/abs/1801.04381) backbone. It uses *depthwise separable convolutions* to achieve strong feature extraction with far fewer parameters, making training significantly faster without sacrificing accuracy.

### Why Binary Cross Entropy?

Road extraction is a binary problem (road or not). Binary Cross Entropy (BCE) loss directly penalizes the predicted probability based on how far it is from the true label:

$$\text{BCE Loss} = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \cdot \log(p_i) + (1 - y_i) \cdot \log(1 - p_i) \right]$$

If a pixel is a road (`y=1`), the loss pushes the prediction `p` toward 1. If it is background (`y=0`), it pushes `p` toward 0.

---

## 2. The Dataset

This tutorial uses the **[DeepGlobe 2018 Road Extraction Challenge](https://doi.org/10.1109/cvprw.2018.00031)** dataset. It contains satellite images of roads across urban, rural, and desert terrain.

Each sample is a pair of files:
- `<id>_sat.jpg` — RGB satellite image (1024×1024)
- `<id>_mask.png` — binary grayscale mask (1024×1024), values `0` or `1`

Since the official test set labels are withheld, we partition the available training data ourselves:

| Split | Fraction |
|---|---|
| Train | 60% |
| Validation | 20% |
| Test | 20% |

The file `metadata_split.csv` records each image's paths and its assigned split. Expected directory layout:

```
data/
  <id>_sat.jpg
  <id>_mask.png
metadata_split.csv
```

> **Note:** Roads typically cover only 5–10% of an image. This class imbalance means a model that predicts "background" everywhere would still score ~95% accuracy — which is why we also track IoU.

---

## 3. Setup

Install the required libraries:

```bash
pip install torch torchvision torchgeo lightning opencv-python pandas matplotlib seaborn
```

Then open `semantic_segmentation.ipynb` and run the cells in order.

---

## Step 1 — Loading the Data

The `RoadDataset` class reads the CSV to find image paths for a given split (`train`, `val`, or `test`), optionally samples a fraction of it (useful for fast experiments), resizes images to 512×512 (the originals are 1024×1024), and converts the mask to a binary float tensor.

**Key points:**
- Images use `BILINEAR` interpolation (smooth) when resizing; masks use `NEAREST` (no interpolation between classes).
- The mask is binarized with `> 0` so any non-zero pixel value is treated as "road".

---

## Step 2 — The Lightning DataModule

The `RoadDataModule` organizes the three dataloaders into one reusable object. PyTorch Lightning calls `setup()` automatically before training starts, instantiating `RoadDataset` for each split and exposing `train_dataloader`, `val_dataloader`, and `test_dataloader`.

---

## Step 3 — Configuring the Model

TorchGeo's `SemanticSegmentationTask` bundles the model, loss function, optimizer, and metrics into a single Lightning module — no boilerplate training loop needed. We configure it with `task="binary"` and `num_classes=1` (a single output channel representing the probability of road), `loss="bce"`, and `weights=True` to load ImageNet-pretrained encoder weights.

The resulting model has **6.6 M trainable parameters** — small enough to train quickly, capable enough to learn road geometry.

---

## Step 4 — Training

Two callbacks are used:
- **`ModelCheckpoint`** — saves only the best model based on `val_loss`
- **`EarlyStopping`** — halts training if validation loss stops improving for 10 consecutive epochs

The `Trainer` automatically uses a GPU if one is available. Training for 20 epochs takes roughly 30 minutes with `batch_size=8` and `img_size=512`.

---

## Step 5 — Evaluation

After training, `trainer.test()` runs the model on the held-out test set and prints all metrics.

### Understanding the Metrics

**Binary Accuracy** — fraction of pixels classified correctly:

$$\text{Accuracy} = \frac{TP + TN}{\text{Total Pixels}}$$

High accuracy alone is misleading on road datasets because background pixels dominate. A model predicting "background" for every pixel would still score ~95%.

**Binary Jaccard Index (IoU)** — the standard segmentation metric. It measures overlap between the predicted and ground-truth road masks, and heavily penalizes both missed roads and false positives:

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$$

A high IoU means the model is accurately capturing the shape and location of road networks.

**Loss (BCE)** — the training objective. Lower is better. Watch that it decreases on both train *and* validation sets; divergence signals overfitting.

---

## Step 6 — Visualizing Predictions

The `visualize_results` function grabs a batch from the test dataloader and plots three columns side by side: the satellite image, the ground truth mask, and the model prediction.

**Note:** the raw model output (`logits`) must be passed through `torch.sigmoid()` to convert to probabilities, then thresholded at 0.5 to get a binary mask.

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 97.59% |
| Test IoU (Jaccard Index) | 52.55% |
| Test Loss (BCE) | 0.069 |

The best checkpoint was saved at **epoch 13** (`val_loss = 0.07`). Training converged smoothly with both loss and IoU improving steadily across 20 epochs.

IoU of ~0.53 is a solid baseline for this dataset and architecture. Common ways to improve it further:
- Train on full 1024×1024 images (currently resized to 512)
- Use the full dataset (currently 50% sampled)
- Add data augmentation (flips, rotations, color jitter)
- Switch to a heavier backbone (e.g., ResNet-50)
- Use a combined BCE + Dice loss

---

## Citation

Demir, I., Koperski, K., Lindenbaum, D., Pang, G., Huang, J., Basu, S., Hughes, F., Tuia, D., & Raskar, R. (2018). DeepGlobe 2018: A Challenge to Parse the Earth through Satellite Images. *CVPR Workshops*. https://doi.org/10.1109/cvprw.2018.00031
