import keras
keras.backend.clear_session()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

print("TF version:", tf.__version__)
np.random.seed(42)
tf.random.set_seed(42)


DATASET_DIR = "blood-cells-image-dataset"
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 20

classes = ["Basophil", "Eosinophil", "Erythroblast", "IG",
           "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"]

class_info = {
    "Basophil": "Rare WBC; involved in allergic reactions",
    "Eosinophil": "WBC; fights parasites and allergens",
    "Erythroblast": "Immature red blood cell precursor",
    "IG": "Immature Granulocyte; sign of infection/inflammation",
    "Lymphocyte": "WBC; key player in adaptive immunity",
    "Monocyte": "Largest WBC; becomes macrophage in tissue",
    "Neutrophil": "Most abundant WBC; first responder to infection",
    "Platelet": "Cell fragment; essential for blood clotting"
}

folders = ["basophil", "eosinophil", "erythroblast", "ig",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]

# collect all image paths
all_paths = []
all_labels = []
for idx, folder in enumerate(folders):
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        for p in (Path(DATASET_DIR) / folder).glob(ext):
            all_paths.append(str(p))
            all_labels.append(idx)

all_paths = np.array(all_paths)
all_labels = np.array(all_labels)
print(f"found {len(all_paths)} images total")

# split into train/val/test (70/15/15)
X_tmp, X_test, y_tmp, y_test = train_test_split(
    all_paths, all_labels, test_size=0.15, random_state=42, stratify=all_labels)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.176, random_state=42, stratify=y_tmp)

print(f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def load_imgs(paths, labels):
    imgs = []
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB").resize(IMG_SIZE)
        img = np.array(img, dtype=np.float32)
        img = preprocess_input(img)
        imgs.append(img)
        if (i + 1) % 2000 == 0:
            print(f"  loaded {i+1}/{len(paths)}")
    return np.array(imgs, dtype=np.float32), np.array(labels)

print("loading images...")
X_tr, y_tr = load_imgs(X_train, y_train)
X_vl, y_vl = load_imgs(X_val, y_val)
X_ts, y_ts = load_imgs(X_test, y_test)
print("done loading")

def denorm(img):
    return np.clip((img + 1.0) / 2.0, 0, 1)

#one sample from each class
samples = {}
for img, lbl in zip(X_tr, y_tr):
    if lbl not in samples:
        samples[lbl] = img
    if len(samples) == 8:
        break

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Blood Cell Types - One Sample per Class", fontsize=15, fontweight="bold")
for i, ax in enumerate(axes.flat):
    ax.imshow(denorm(samples[i]))
    ax.set_title(f"{classes[i]}\n{class_info[classes[i]]}", fontsize=8, fontweight="bold")
    ax.axis("off")
plt.tight_layout()
plt.savefig("graph1_sample_images.png", dpi=150, bbox_inches="tight")
plt.show()
print("graph 1 done")

#class distribution
counts = [int((y_tr == i).sum()) for i in range(8)]
colors = plt.cm.Set2.colors

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Class Distribution - Training Set", fontsize=14, fontweight="bold")

bars = axes[0].bar(classes, counts, color=colors, edgecolor="black", linewidth=0.5)
axes[0].set_ylabel("Number of Images")
axes[0].set_title("Counts per Class")
axes[0].tick_params(axis="x", rotation=35)
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 10, str(count), ha="center", fontsize=9)

axes[1].pie(counts, labels=classes, colors=colors, autopct="%1.1f%%",
            startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 1.2})
axes[1].set_title("Proportion per Class")

plt.tight_layout()
plt.savefig("graph2_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("graph 2 done")

#build model using mobilenetv2 as base
base = keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False
print(f"base model has {len(base.layers)} layers, all frozen")

inp = keras.Input(shape=(*IMG_SIZE, 3), name="image_input")
x = base(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(8, activation="softmax")(x)
model = keras.Model(inp, out, name="BloodCell_CNN")
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# phase 1: train only the head
print("\nphase 1 - training head only")
history1 = model.fit(
    X_tr, y_tr,
    validation_data=(X_vl, y_vl),
    epochs=5,
    batch_size=BATCH_SIZE,
    verbose=1
)

# phase 2: unfreeze last 30 layers and fine-tune
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

n_trainable = sum(1 for l in base.layers if l.trainable)
print(f"\nphase 2 - fine-tuning {n_trainable} layers with lower lr")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cbs = [
    EarlyStopping(monitor="val_accuracy", patience=7,
                  restore_best_weights=True, mode="max", verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                     patience=3, min_lr=1e-7, verbose=1)
]

history2 = model.fit(
    X_tr, y_tr,
    validation_data=(X_vl, y_vl),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cbs,
    verbose=1
)

# merge both training histories
acc = history1.history["accuracy"] + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss = history1.history["loss"] + history2.history["loss"]
val_loss = history1.history["val_loss"] + history2.history["val_loss"]

# evaluate on test set
test_loss, test_acc = model.evaluate(X_ts, y_ts, verbose=0)
print(f"\ntest accuracy: {test_acc * 100:.2f}%")
print(f"test loss: {test_loss:.4f}")

y_probs = model.predict(X_ts, verbose=0)
y_pred = np.argmax(y_probs, axis=1)
print("\nclassification report:")
print(classification_report(y_ts, y_pred, target_names=classes))

# graph 3 - training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Training History (Phase 1 + Fine-tuning)", fontsize=14, fontweight="bold")
ep = range(1, len(acc) + 1)
p1 = len(history1.history["accuracy"])

ax1.plot(ep, acc, label="Train", linewidth=2)
ax1.plot(ep, val_acc, label="Val", linewidth=2, linestyle="--")
ax1.axvline(x=p1, color="gray", linestyle=":", linewidth=1.5, label="fine-tune start")
ax1.set_title("Accuracy")
ax1.set_xlabel("Epoch")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(ep, loss, label="Train", linewidth=2)
ax2.plot(ep, val_loss, label="Val", linewidth=2, linestyle="--")
ax2.axvline(x=p1, color="gray", linestyle=":", linewidth=1.5, label="fine-tune start")
ax2.set_title("Loss")
ax2.set_xlabel("Epoch")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("graph3_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("graph 3 done")

# graph 4 - confusion matrix
cm = confusion_matrix(y_ts, y_pred)
cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold")

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes,
            ax=axes[0], linewidths=0.4)
axes[0].set_title("Raw Counts")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")
axes[0].tick_params(axis="x", rotation=40)

sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=classes, yticklabels=classes,
            ax=axes[1], linewidths=0.4)
axes[1].set_title("Percentage per True Class (%)")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")
axes[1].tick_params(axis="x", rotation=40)

plt.tight_layout()
plt.savefig("graph4_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("graph 4 done")

#class accuracy bar chart
per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
bar_colors = ["#2ecc71" if a >= 85 else "#e67e22" if a >= 70 else "#e74c3c"
              for a in per_class_acc]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(classes, per_class_acc, color=bar_colors, edgecolor="black", linewidth=0.5)
ax.axvline(x=test_acc * 100, color="navy", linestyle="--", linewidth=1.5,
           label=f"overall: {test_acc * 100:.1f}%")
ax.set_xlabel("Accuracy (%)")
ax.set_title("Per-Class Accuracy on Test Set", fontsize=13, fontweight="bold")
ax.set_xlim(0, 115)
for bar, a in zip(bars, per_class_acc):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{a:.1f}%", va="center", fontsize=9)

legend_patches = [
    mpatches.Patch(color="#2ecc71", label=">= 85% (Excellent)"),
    mpatches.Patch(color="#e67e22", label="70-85% (Good)"),
    mpatches.Patch(color="#e74c3c", label="< 70% (Needs work)")
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("graph5_per_class_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()
print("graph 5 done")

#misclassified examples
wrong = np.where(y_pred != y_ts)[0][:10]
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Misclassified Examples", fontsize=13, fontweight="bold")
for ax, idx in zip(axes.flat, wrong):
    ax.imshow(denorm(X_ts[idx]))
    ax.set_title(f"True: {classes[y_ts[idx]]}\nPred: {classes[y_pred[idx]]} ({y_probs[idx][y_pred[idx]]*100:.0f}%)",
                 fontsize=8, color="red")
    ax.axis("off")
plt.tight_layout()
plt.savefig("graph6_misclassified.png", dpi=150, bbox_inches="tight")
plt.show()
print("graph 6 done")

#distribution correct vs wrong
correct_conf = y_probs[np.arange(len(y_ts)), y_pred][y_pred == y_ts] * 100
wrong_conf = y_probs[np.arange(len(y_ts)), y_pred][y_pred != y_ts] * 100

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(correct_conf, bins=40, alpha=0.65, color="#2ecc71",
        label=f"Correct ({len(correct_conf)})", edgecolor="none", density=True)
ax.hist(wrong_conf, bins=40, alpha=0.65, color="#e74c3c",
        label=f"Wrong ({len(wrong_conf)})", edgecolor="none", density=True)
ax.axvline(x=50, color="black", linestyle="--", linewidth=1.2, label="50% threshold")
ax.set_xlabel("Prediction Confidence (%)")
ax.set_ylabel("Density")
ax.set_title("Model Confidence: Correct vs Wrong", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graph7_confidence_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("graph 7 done")

#feature maps from first conv layer
mobilenet_sub = model.get_layer("mobilenetv2_1.00_96")
feature_model = keras.Model(
    inputs=mobilenet_sub.input,
    outputs=mobilenet_sub.get_layer("Conv1").output
)

sample = X_ts[0]
fmaps = feature_model.predict(tf.expand_dims(sample, 0), verbose=0)[0]
n = min(31, fmaps.shape[-1])

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle(f"Conv1 Feature Maps - Input: {classes[y_ts[0]]}", fontsize=13, fontweight="bold")
axes[0, 0].imshow(denorm(sample))
axes[0, 0].set_title("Input", fontsize=8)
axes[0, 0].axis("off")

for i in range(1, n + 1):
    r, c = divmod(i, 8)
    axes[r, c].imshow(fmaps[:, :, i - 1], cmap="viridis")
    axes[r, c].set_title(f"Filter {i}", fontsize=7)
    axes[r, c].axis("off")

for i in range(n + 1, 32):
    r, c = divmod(i, 8)
    axes[r, c].set_visible(False)

plt.tight_layout()
plt.savefig("graph8_feature_maps.png", dpi=150, bbox_inches="tight")
plt.show()
print("graph 8 done")

model.save("blood_cell_cnn_model.keras")
print(f"\ndone! test accuracy: {test_acc * 100:.2f}%")
