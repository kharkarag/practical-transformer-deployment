#!/bin/bash
gsutil cp -r gs://project-models/gdrive .
python convert_graph_to_onnx.py --check-loading --pipeline question-answering --framework pt --model gdrive/models/bert-base-trained --quantize bert-base-quant/bert-base.quant.onnx
python convert_graph_to_onnx.py --check-loading --pipeline question-answering --framework pt --model gdrive/models/bert-tiny-trained --quantize bert-tiny-quant/bert-tiny.quant.onnx
python convert_graph_to_onnx.py --check-loading --pipeline question-answering --framework pt --model gdrive/models/distilbert-base-trained --quantize distilbert-quant/distilbert.quant.onnx
python convert_graph_to_onnx.py --check-loading --pipeline question-answering --framework pt --model gdrive/models/electra-small-trained --quantize electra-small-quant/electra-small.quant.onnx