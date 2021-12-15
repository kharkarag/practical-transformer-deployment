#!/bin/bash
gsutil cp -r gs://project-models/gdrive .
python convert_graph_to_onnx.py --check-loading --pipeline question-answering --framework pt --model gdrive/models/bert-base-trained bert-base/bert-base.onnx
python convert_graph_to_onnx.py --check-loading --pipeline question-answering --framework pt --model gdrive/models/bert-tiny-trained bert-tiny/bert-tiny.onnx
python convert_graph_to_onnx.py --check-loading --pipeline question-answering --framework pt --model gdrive/models/distilbert-base-trained distilbert/distilbert.onnx
python convert_graph_to_onnx.py --check-loading --pipeline question-answering --framework pt --model gdrive/models/electra-small-trained electra-small/electra-small.onnx
