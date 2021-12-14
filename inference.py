from flask import Flask, request, abort
app = Flask(__name__)
import time
from transformers import DistilBertTokenizerFast, BertTokenizerFast, ElectraTokenizerFast

# Global variables
tokenizer, ort_session, model_path = None, None, None
supported_models = ['bert', 'bert-small', 'bert-tiny', 'distilbert']

# Imports and env vars for onnx
from os import environ
from psutil import cpu_count

import onnxruntime as ort
from onnxruntime import InferenceSession, SessionOptions
print(f"Running on {ort.get_device()}")

def create_model_for_provider(model_path: str, num_threads: int = 1) -> InferenceSession:
    """
    Creates an ONNX inference session for the specified model and hardware type.
    Args:
        model_path (str): filepath to the model on disk
        num_threads (int): number of intra-op threads for session
    Returns:
        (InferenceSession): ONNX Runtime inference session
    """
    # Few properties than might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = num_threads
    providers = ['CPUExecutionProvider'] if ort.get_device() == 'CPU' else ['CUDAExecutionProvider']

    # Load the model as a graph and prepare the CPU backend
    return InferenceSession(model_path, options, providers=providers)


@app.route("/set_model", methods=['POST', 'GET'])
def set_model() -> str:
    """
    Loads the specified model type for future inference requests.
    Request args:
        model_type (str): model to load. Must be in ['bert-base', 'bert-tiny', 'distilbert', 'electra-small']
        use_onnx_optim (bool): whether to use ONNX-optimized model
        num_threads (int): number of intra-op threads for session 
    Returns:
        (str): success message
    """
    global tokenizer, ort_session, model_path

    if request.method == 'GET':
        return "GET not supported", 404

    # Parse request args
    model_type = request.args.get("model_type")
    use_onnx_quant = request.args.get("use_onnx_quant", "False") == "True"
    num_threads = int(request.args.get("num_threads", 1))
    if model_type is None:
        return "Requests to `set_model` must include `model_type`", 400

    # Set tokenizer
    if 'distil' in model_type:
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    elif 'bert' in model_type:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    elif 'electra' in model_type:
        tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')

    # Set model path
    if use_onnx_quant:
        # model_path = f"models/{model_type}.quant.onnx"
        model_path = f"{model_type}-quant/{model_type}.onnx"
    else:
        # model_path = f"models/{model_type}.onnx"
        model_path = f"{model_type}/{model_type}.onnx"

    ort_session = create_model_for_provider(model_path, num_threads)

    print(f"Model set to {model_type} at {model_path}")
    return f"Model set to {model_type}"


@app.route("/inference", methods=['POST', 'GET'])
def inference() -> dict:
    """
    Performs inference on the previously loaded model with the provided batch of sentences.
    Request args:
        [json] (str): JSON parameter of request must contain list of sentences
    Returns:
        (dict): inference metrics
    """
    global tokenizer, ort_session

    if request.method == 'GET':
        return "GET not supported", 404
    if tokenizer is None or ort_session is None:
        return f"Tokenizer and model are not set. Please call the `set_model` URL with argument `model_type` in {supported_models}", 500

    # Parse request args
    input_strings = request.get_json()
    if input_strings is None or len(input_strings) < 1:
        return "Requests to `inference` must include `input_string`", 400

    # Perform inference
    start_time = time.time()

    inputs = tokenizer(input_strings, return_tensors="np")
    # outputs = ort_session.run(["last_hidden_state"], dict(inputs))
    outputs = ort_session.run(["output_0"], dict(inputs))
    predictions = outputs[0].tolist()

    end_time = time.time()
    total_time = end_time - start_time
    input_size = len(inputs)

    return {"predictions": predictions, "time": total_time, "input_size": input_size, "model_path": model_path}
