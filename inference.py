from flask import Flask
from flask import request
app = Flask(__name__)

from torch.nn.functional import softmax
import onnxruntime as ort


@app.route("/inference", methods=['POST', 'GET'])
def inference():
    if request.method == 'GET':
        return "GET not supported"
    
    model_type = request.args.get("model_type")
    input_string = request.args.get("input_string")

    onnx_path = f"models/{model_type}.onnx"
    ort_session = ort.InferenceSession(onnx_path)
    
    inputs = tokenizer(input_string, return_tensors="np")
    outputs = ort_session.run(["last_hidden_state"], dict(inputs))
    predictions = softmax(outputs[0])

    return {"predictions": predictions}
