from flask import Flask, request, abort
app = Flask(__name__)

from torch.nn.functional import softmax
import onnxruntime as ort

# Global variables
tokenizer, ort_session = None, None
supported_models = ['bert', 'bert-small', 'bert-tiny', 'distilbert']


@app.route("/set_model", methods=['POST', 'GET'])
def set_model():
    global tokenizer, ort_session

    if request.method == 'GET':
        return "GET not supported", 404

    model_type = request.args.get("model_type")
    use_onnx_optim = request.args.get("use_onnx_optim", False)
    if model_type is None:
        return "Requests to `set_model` must include `model_type`", 400

    if 'distil' in model_type:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    elif 'bert' in model_type:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if use_onnx_optim:
        onnx_path = f"models/{model_type}_optim.onnx"
    else:
        onnx_path = f"models/{model_type}.onnx"
    ort_session = ort.InferenceSession(onnx_path)
        

@app.route("/inference", methods=['POST', 'GET'])
def inference():
    if request.method == 'GET':
        return "GET not supported", 404
    if tokenizer is None or ort_session is None:
        return f"Tokenizer and model are not set. Please call the `set_model` URL with argument `model_type` in {supported_models}", 500

    input_strings = request.get_json()
    if input_strings is None or len(input_strings) < 1:
        return "Requests to `inference` must include `input_string`", 400
    
    inputs = tokenizer(input_strings, return_tensors="np")
    outputs = ort_session.run(["last_hidden_state"], dict(inputs))
    predictions = softmax(outputs[0])

    return {"predictions": predictions}
