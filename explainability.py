# ===============================
# Explainable AI for Healthcare Imaging
# CNN + Grad-CAM (+ optional LIME, SHAP)
# ===============================

import os
import random
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# -------------------------------
# 0) Repro + TF GPU-safe settings
# -------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("GPU memory growth could not be set:", e)

# -------------------------------
# 1) Paths & Config
# -------------------------------
# Change this to your dataset root (folder containing train/val/test)
DATA_DIR = r"./chest_xray"     # <-- EDIT THIS PATH
OUTPUT_DIR = "./outputs_explainable_ai"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5                     # Keep small for demo; increase for better accuracy
CLASS_MODE = "binary"          # NORMAL vs PNEUMONIA

# -------------------------------
# 2) Data pipeline
# -------------------------------
# Train-time augmentation (crucial for medical images to reduce overfitting)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    shuffle=True,
    seed=SEED
)
val_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    shuffle=False
)
test_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    shuffle=False
)

class_indices = train_gen.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}
print("Class mapping:", idx_to_class)

# -------------------------------
# 3) Model: Transfer Learning (MobileNetV2)
# -------------------------------
base = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base.trainable = False  # freeze for fast training

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)  # binary
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

model.summary()

# -------------------------------
# 4) Train
# -------------------------------
ckpt_path = os.path.join(OUTPUT_DIR, "best_model.keras")
ckpt = keras.callbacks.ModelCheckpoint(
    ckpt_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1
)
early = keras.callbacks.EarlyStopping(
    monitor="val_auc", mode="max", patience=3, restore_best_weights=True
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[ckpt, early],
    verbose=1
)

# Save final model
final_model_path = os.path.join(OUTPUT_DIR, "final_model.keras")
model.save(final_model_path)
print(f"Saved model to {final_model_path}")

# -------------------------------
# 5) Evaluate
# -------------------------------
eval_res = model.evaluate(test_gen, verbose=1)
print("Test set evaluation:", dict(zip(model.metrics_names, eval_res)))

# -------------------------------
# 6) Utility: show image + prediction
# -------------------------------
def load_random_test_image():
    """Returns (img_array_preprocessed, raw_img_0_255, label_index, file_path)."""
    # pick random test image
    filepaths = test_gen.filepaths
    fp = random.choice(filepaths)
    img = keras.utils.load_img(fp, target_size=IMG_SIZE)
    arr = keras.utils.img_to_array(img)
    raw = arr.copy().astype("uint8")  # for visualization
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    # label from filenames via generator (tedious) -> use folder name
    label_name = os.path.basename(os.path.dirname(fp))
    label_idx = class_indices[label_name]
    return arr, raw, label_idx, fp

def predict_and_print(arr, label_idx, raw_img=None, filepath=None):
    prob = float(model.predict(arr, verbose=0)[0][0])
    pred_label_idx = int(prob >= 0.5)
    print("\n--- Prediction ---")
    if filepath:
        print("File:", filepath)
    print(f"True label:     {idx_to_class[label_idx]} ({label_idx})")
    print(f"Pred prob[1]:   {prob:.4f}  -> Pred class: {idx_to_class[pred_label_idx]}")

    if raw_img is not None:
        plt.figure(figsize=(4,4))
        plt.imshow(raw_img.astype("uint8"))
        plt.axis("off")
        plt.title(f"Pred: {idx_to_class[pred_label_idx]} (p={prob:.2f})\nTrue: {idx_to_class[label_idx]}")
        plt.tight_layout()
        outp = os.path.join(OUTPUT_DIR, "sample_prediction.png")
        plt.savefig(outp, dpi=160)
        print("Saved:", outp)
        plt.show()

# quick sanity check on one image
arr, raw, y_true, fp = load_random_test_image()
predict_and_print(arr, y_true, raw_img=raw, filepath=fp)

# -------------------------------
# 7) Grad-CAM
# -------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    - img_array: preprocessed (1, H, W, 3)
    - model: Keras model with a conv backbone
    - last_conv_layer_name: if None, try to find last Conv2D layer
    """
    # Autodetect last conv layer if not provided
    if last_conv_layer_name is None:
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, layers.Conv2D):
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            # If base is wrapped, find inside base model
            for layer in reversed(model.layers):
                if hasattr(layer, 'layers'):
                    for l2 in reversed(layer.layers):
                        if isinstance(l2, layers.Conv2D):
                            last_conv_layer_name = l2.name
                            break
                if last_conv_layer_name is not None:
                    break

    # Build grad-model mapping input to last conv outputs and predictions
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        (conv_outputs, predictions) = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        target = predictions[:, pred_index]

    # Gradients of target wrt conv outputs
    grads = tape.gradient(target, conv_outputs)
    # Channel-wise average of gradients
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    # Weighted sum of conv maps
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)

    # Normalize to [0, 1]
    heatmap = tf.maximum(cam, 0) / tf.reduce_max(cam + 1e-10)
    return heatmap.numpy()

def overlay_gradcam(heatmap, image_uint8, alpha=0.4, cmap='jet'):
    import cv2
    heatmap_resized = cv2.resize(heatmap, (image_uint8.shape[1], image_uint8.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_uint8, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# Generate Grad-CAM on a few samples
for i in range(3):
    arr, raw, y_true, fp = load_random_test_image()
    heatmap = make_gradcam_heatmap(arr, model, last_conv_layer_name=None)
    try:
        import cv2
        overlay = overlay_gradcam(heatmap, raw)
        plt.figure(figsize=(9,3))
        plt.subplot(1,3,1); plt.imshow(raw); plt.axis("off"); plt.title("Original")
        plt.subplot(1,3,2); plt.imshow(heatmap, cmap='jet'); plt.axis("off"); plt.title("Grad-CAM heatmap")
        plt.subplot(1,3,3); plt.imshow(overlay); plt.axis("off"); plt.title("Overlay")
        plt.tight_layout()
        outp = os.path.join(OUTPUT_DIR, f"gradcam_{i}.png")
        plt.savefig(outp, dpi=160)
        print("Saved Grad-CAM:", outp)
        plt.show()
    except ImportError:
        print("OpenCV not installed. Install with: pip install opencv-python")

# -------------------------------
# 8) (Optional) LIME explanations
# -------------------------------
# Install if needed: pip install lime scikit-image
def run_lime_example(n_samples=1):
    try:
        from lime import lime_image
        from skimage.segmentation import quickshift
    except ImportError:
        print("LIME or scikit-image not installed. Install with: pip install lime scikit-image")
        return

    # LIME needs a predictor that takes a batch of RGB [0..255] and returns probs
    def lime_predict(batch_imgs):
        # batch_imgs shape: (N, H, W, 3) in 0..255 float
        batch = preprocess_input(batch_imgs.copy())
        probs = model.predict(batch, verbose=0)
        # LIME expects 2-class probabilities; build [p0, p1]
        probs = np.hstack([1 - probs, probs])
        return probs

    explainer = lime_image.LimeImageExplainer()

    for i in range(n_samples):
        arr, raw, y_true, fp = load_random_test_image()
        img = raw.astype(np.uint8)

        explanation = explainer.explain_instance(
            image=img,
            classifier_fn=lime_predict,
            top_labels=1,
            hide_color=0,
            num_samples=1000,
            segmentation_fn=lambda x: quickshift(x, kernel_size=3, max_dist=6, ratio=0.5)
        )
        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            label=top_label,
            positive_only=True,
            num_features=8,
            hide_rest=False
        )
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1); plt.imshow(img); plt.axis("off"); plt.title(f"Original\nTrue: {idx_to_class[y_true]}")
        plt.subplot(1,2,2); plt.imshow(temp); plt.axis("off"); plt.title("LIME: Important regions")
        plt.tight_layout()
        outp = os.path.join(OUTPUT_DIR, f"lime_{i}.png")
        plt.savefig(outp, dpi=160)
        print("Saved LIME:", outp)
        plt.show()

# Uncomment to run LIME demo (requires extra installs)
# run_lime_example(n_samples=2)

# -------------------------------
# 9) (Optional) SHAP explanations
# -------------------------------
# Install if needed: pip install shap
def run_shap_example(n_background=50, n_explain=5):
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Install with: pip install shap")
        return

    # Build small background set from validation
    background = []
    for i in range(n_background):
        arr, raw, _, _ = load_random_test_image()
        background.append(arr[0])
    background = np.stack(background, axis=0)

    explainer = shap.GradientExplainer(model, background)

    # Pick a few test images to explain
    explain_set = []
    raws = []
    labels = []
    for i in range(n_explain):
        arr, raw, y_true, _ = load_random_test_image()
        explain_set.append(arr[0])
        raws.append(raw)
        labels.append(y_true)
    explain_set = np.stack(explain_set, axis=0)

    shap_values = explainer.shap_values(explain_set)

    # shap_values is a list for each output (binary -> 1 array). Take class 1.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Plot SHAP for first image (expected pixel importance)
    for i in range(len(explain_set)):
        plt.figure(figsize=(5,5))
        shap.image_plot([shap_values[i]], -explain_set[i], show=False)
        plt.title(f"SHAP (class=1) — True: {idx_to_class[labels[i]]}")
        outp = os.path.join(OUTPUT_DIR, f"shap_{i}.png")
        plt.savefig(outp, dpi=160, bbox_inches="tight")
        print("Saved SHAP:", outp)
        plt.show()

# Uncomment to run SHAP demo (requires extra install and can be slow)
# run_shap_example(n_background=30, n_explain=2)

print("\nDone. Model + Grad-CAM outputs are in:", OUTPUT_DIR)
print("Optional: uncomment run_lime_example() / run_shap_example() to generate LIME/SHAP visualizations.")

