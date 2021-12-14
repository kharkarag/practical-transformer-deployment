import os
import argparse
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    args = parser.parse_args()

    model_path = Path(args.model_path)
    for model_file in model_path.iterdir():
        if len(model_file.suffixes) == 1 and model_file.suffixes[-1] == '.onnx' and '-opt' not in str(model_file):
            print(f"Quantizing {model_file}")
            model_quant = model_file.with_suffix('.quant.onnx')
            quantized_model = quantize_dynamic(model_file, model_quant, weight_type=QuantType.QInt8)
