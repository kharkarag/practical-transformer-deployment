from transformers.onnx import OnnxConfig
from transformers import TensorType, BertPreTrainedModel, DistilBertPreTrainedModel
from collections import OrderedDict
from typing import Mapping
from itertools import chain
import argparse

class BertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
                ("token_type_ids", {0: "batch", 1: "sequence"}),
            ]
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([("last_hidden_state", {0: "batch", 1: "sequence"}), ("pooler_output", {0: "batch"})])


class DistilBertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
            ]
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([("last_hidden_state", {0: "batch", 1: "sequence"})])


def to_onnx(model, tokenizer, out_path):
    if isinstance(model, BertPreTrainedModel):
        config = BertOnnxConfig(model.config)
    elif isinstance(model, DistilBertPreTrainedModel):
        config = DistilBertOnnxConfig(model.config)
    
    dummy_inputs = config.generate_dummy_inputs(tokenizer, framework=TensorType.PYTORCH)
    
    torch.onnx.export(model,
                     (dummy_inputs,),
                     f=out_path,
                     input_names=list(config.inputs.keys()),
                     output_names=list(config.outputs.keys()),
                     dynamic_axes={name: axes for name, axes in chain(config.inputs.items(), config.outputs.items())},
                     do_constant_folding=True,
                     use_external_data_format=config.use_external_data_format(model.num_parameters()),
                     enable_onnx_checker=True,
                     opset_version=12,
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    args = parser.parse_args()

    tokenizer = DistilBertTokenizer.from_pretrained(args.input_path)
    model = DistilBertForSequenceClassification.from_pretrained(args.input_path)

    to_onnx(model, tokenizer, args.out_path)
