# Heart Reader Challenge — ECG Multi-Label Classifier

Automated multi-label diagnostic classification of 12-lead ECGs into 5 cardiac superclasses using a 1-D ResNet trained on PTB-XL v1.0.3.

## Results

| Metric | Score |
|---|---|
| **Test Macro F1** | **0.6741** |
| **Test Macro AUC** | **0.8822** |
| Optimal Threshold | 0.60 |
| Training Epochs | 28 (early stopping, best at epoch 13) |

### Per-Class Performance (Test Set, fold 10)

| Class | Description | F1 | AUC |
|---|---|---|---|
| NORM | Normal ECG | 0.839 | 0.928 |
| MI | Myocardial Infarction | 0.698 | 0.890 |
| STTC | ST/T-wave Changes | 0.713 | 0.914 |
| CD | Conduction Disturbance | 0.721 | 0.892 |
| HYP | Hypertrophy | 0.400 | 0.787 |

> HYP has the lowest F1 due to its small sample size and morphological overlap with other classes.

## Model Architecture

**ResNet1D** — 1D Residual Convolutional Neural Network

```
Input: (B, 12, 1000)  ← 12-lead ECG, 1000 samples @ 100 Hz
  │
  ├─ Stem: Conv1d(12→64, k=15) → BN → ReLU → MaxPool
  ├─ Layer1: ResBlock(64→128, stride=2)
  ├─ Layer2: ResBlock(128→256, stride=2)
  ├─ Layer3: ResBlock(256→256, stride=2)
  ├─ Layer4: ResBlock(256→512, stride=2)
  ├─ Global Average Pooling
  ├─ Dropout(0.3)
  ├─ FC(512→128) → ReLU
  └─ FC(128→5) → Sigmoid

Output: (B, 5)  ← Multi-label probabilities
```

**Training details:**
- Loss: BCEWithLogitsLoss with class-frequency-weighted `pos_weight`
- Optimiser: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Mixed precision (AMP), trained on Apple Silicon MPS
- Early stopping (patience=15) on validation macro-F1
- Post-training threshold optimisation on fold 9 → best threshold=0.60

## Superclasses

| Class | Description |
|---|---|
| NORM | Normal |
| MI | Myocardial Infarction |
| STTC | ST/T Change |
| CD | Conduction Disturbance |
| HYP | Hypertrophy |

## Project Structure

```
├── config.py               # Central hyperparameter config
├── train.py                # Training entry-point
├── evaluate.py             # Test-set evaluation (fold 10)
├── inference.py            # Single-record inference demo
├── optimize.py             # ONNX / TorchScript / INT8 export
├── visualize_results.py    # ROC curves + confusion matrices
├── download_data.py        # Dataset downloader
├── Makefile                # Convenience commands
├── requirements.txt
│
├── src/
│   ├── data/
│   │   └── dataset.py      # PTBXLDataset + label mapping + augmentation
│   ├── models/
│   │   └── resnet1d.py     # ResNet1D architecture
│   └── utils/
│       ├── metrics.py      # Macro F1, AUC, threshold search
│       └── trainer.py      # Train/eval loops + EarlyStopping
│
├── checkpoints/
│   └── best_model.pt       # Best checkpoint (val F1=0.6840, threshold=0.60)
│
└── results/
    ├── model.onnx           # ONNX FP32 export (~38 MB)
    ├── model_scripted.pt    # TorchScript FP32 (~38 MB)
    ├── model_int8_scripted.pt  # TorchScript INT8 (dynamic quantized)
    ├── model.tflite         # TFLite INT8 for mobile/embedded deployment
    ├── roc_curves.png       # ROC curves for all 5 classes
    ├── confusion_matrices.png  # Per-class confusion matrices
    └── test_results.json    # Full metrics in JSON
```

## Data Split

| Split | Folds | Samples |
|---|---|---|
| Train | 1 – 8 | ~17,441 |
| Val | 9 | ~2,174 |
| Test | 10 | ~2,174 |

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
```bash
python download_data.py
```
This downloads PTB-XL v1.0.3 (~1.7 GB) into `./ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/`.

### 3. Train
```bash
python train.py
```

### 4. Evaluate on the test set (fold 10)
```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
```

### 5. Visualise results
```bash
python visualize_results.py
```

### 6. Edge optimisation
```bash
python optimize.py --checkpoint checkpoints/best_model.pt --quantize
```

### 7. Run inference on a single file
```bash
python inference.py --file path/to/ecg_record --checkpoint checkpoints/best_model.pt
```

## Edge Deployment

### Exported Model Formats

| Format | File | Notes |
|---|---|---|
| TorchScript FP32 | `results/model_scripted.pt` | Full precision, any PyTorch runtime |
| ONNX FP32 | `results/model.onnx` | Cross-framework, ONNX Runtime |
| TorchScript INT8 | `results/model_int8_scripted.pt` | Dynamic quantized (Linear layers, qnnpack) |
| TFLite INT8 | `results/model.tflite` | Mobile/embedded (Raspberry Pi, Android) |

### ONNX Runtime inference
```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("results/model.onnx")
ecg = np.random.randn(1, 12, 1000).astype(np.float32)
logits = sess.run(["logits"], {"ecg": ecg})[0]
probs = 1 / (1 + np.exp(-logits))  # sigmoid
# probs shape: (1, 5) → [NORM, MI, STTC, CD, HYP]
```

### TFLite inference
```python
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="results/model.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

ecg = np.random.randn(1, 12, 1000).astype(np.float32)
interpreter.set_tensor(inp['index'], ecg)
interpreter.invoke()
probs = interpreter.get_tensor(out['index'])
# probs shape: (1, 5) → [NORM, MI, STTC, CD, HYP]
```

### Convert ONNX → TFLite (reproduce)
```python
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

model = onnx.load("results/model.onnx")
tf_rep = prepare(model)
tf_rep.export_graph("results/tf_saved_model")

converter = tf.lite.TFLiteConverter.from_saved_model("results/tf_saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("results/model.tflite", "wb") as f:
    f.write(tflite_model)
```

## Reproducibility

All experiments use `seed=42`. The PTB-XL 10-fold split is deterministic.

```bash
python train.py     # trains from scratch → checkpoints/best_model.pt
python evaluate.py  # should reproduce Macro F1 ≈ 0.674, AUC ≈ 0.882
```

## License

Code: MIT License.
Dataset: PTB-XL is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) via PhysioNet.
