"""
Edge optimisation – Export the trained model to TorchScript and ONNX,
then optionally quantise to INT8.

The exported ONNX model can be converted to TFLite via:
    pip install onnx-tf tensorflow
    python -c "
    import onnx
    from onnx_tf.backend import prepare
    model = onnx.load('results/model.onnx')
    tf_rep = prepare(model)
    tf_rep.export_graph('results/tf_model')
    "
    # Then use TFLite converter from the SavedModel.

Usage:
    python optimize.py --checkpoint checkpoints/best_model.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from config import CFG
from src.models.resnet1d import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="./checkpoints/best_model.pt")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--quantize",   action="store_true",
                   help="Apply dynamic INT8 quantisation (CPU only).")
    return p.parse_args()


def export_torchscript(model: nn.Module, dummy: torch.Tensor, path: Path):
    scripted = torch.jit.trace(model, dummy)
    scripted.save(str(path))
    print(f"TorchScript saved → {path}")


def export_onnx(model: nn.Module, dummy: torch.Tensor, path: Path):
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["ecg"],
        output_names=["logits"],
        dynamic_axes={"ecg": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )
    print(f"ONNX model saved → {path}")


def quantize_dynamic(model: nn.Module) -> nn.Module:
    # Dynamic quantisation is only reliably supported for nn.Linear on CPU/ARM.
    # Conv1d INT8 requires FBGEMM/QNNPACK which is unavailable on MPS.
    import torch.backends
    if hasattr(torch.backends, "quantized"):
        torch.backends.quantized.engine = "qnnpack"  # ARM/Apple Silicon
    from torch.ao.quantization import quantize_dynamic as _qd
    q_model = _qd(model, {nn.Linear}, dtype=torch.qint8)
    print("Dynamic INT8 quantisation applied (Linear layers).")
    return q_model


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(num_classes=CFG.num_classes, dropout=0.0)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(1, 12, 1000)   # (batch=1, leads=12, time=1000)

    # ── Standard exports ──────────────────────────────────────────────────────
    export_torchscript(model, dummy, out / "model_scripted.pt")
    export_onnx(model, dummy, out / "model.onnx")

    # ── Optional quantisation ─────────────────────────────────────────────────
    if args.quantize:
        q_model = quantize_dynamic(model)
        # TorchScript export works fine for quantized models.
        # ONNX export of quantized packed params is not yet supported by the
        # dynamo exporter in PyTorch 2.10 — skip it for the INT8 model.
        export_torchscript(q_model, dummy, out / "model_int8_scripted.pt")

        # Report size reduction
        orig_size = (out / "model_scripted.pt").stat().st_size / 1e6
        q_size    = (out / "model_int8_scripted.pt").stat().st_size / 1e6
        print(f"\nSize: FP32={orig_size:.1f} MB → INT8={q_size:.1f} MB "
              f"({(1 - q_size / orig_size) * 100:.1f}% reduction)")

    print("\nOptimisation complete.")
    print("To convert to TFLite, install onnx-tf and tensorflow, then run:")
    print("  python -c \"import onnx; from onnx_tf.backend import prepare; "
          "m=onnx.load('results/model.onnx'); tf_rep=prepare(m); "
          "tf_rep.export_graph('results/tf_saved_model')\"")
    print("Then use tf.lite.TFLiteConverter.from_saved_model('results/tf_saved_model').")


if __name__ == "__main__":
    main()
