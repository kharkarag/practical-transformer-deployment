import numpy as np
from flask import Flask, request, abort, render_template
import time
import requests
from scipy.special import softmax
from transformers import DistilBertTokenizerFast, BertTokenizerFast, ElectraTokenizerFast

from collections import defaultdict
from datasets import load_dataset, load_metric

# Globals
SENTENCES = 250
sentence_table_log, trial, data, model_path, model_metrics = None, None, None, None, None
supported_models = ['bert', 'bert-small', 'bert-tiny', 'distilbert']
model_types = ['bert-base', 'distilbert', 'electra-small', 'bert-tiny']
first_start = True

models = {}
tokenizers = {}

app = Flask(__name__)

import time
from transformers import DistilBertTokenizerFast, BertTokenizerFast, ElectraTokenizerFast

# Imports and env vars for onnx
from os import environ
from psutil import cpu_count

import onnxruntime as ort
from onnxruntime import InferenceSession, SessionOptions


def initialize_tables():
    global model_metrics, sentence_table_log, trial

    # reset other globals
    trial = 0

    # sentence table log
    sentence_table_log = []

    # model metrics table
    model_metrics = defaultdict(dict)
    for model in model_types:
        model_metrics[model] = {
            'acc' : [],
            'speedup' : [],
        }



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





## FIXME: NEED TO UPDATE THIS FOR NEW DATA
def inference_request(url: str, batch_size: int = 1, seq_multiplier: int = 1):
    """
    Sends an inference request with a batch of dummy sentences.
    Args:
        url (str): URL of the endpoint to send the request
        batch_size (int): number of dummy sentences in the batch
    Returns:
        (request.Response): response from the endpoint
        (float): time of request
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = None
    # Time request
    start_time = time.time()
    response = requests.post(url, json=data, headers=headers)
    end_time = time.time()

    return response, end_time - start_time




def get_url(local: bool, remote_type: str = None) -> str:
    """
    Maps the locality and type of endpoint to a URL for an endpoint.
    Args:
        local (bool): whether to access the local endpoint or a remote one
        remote_type (str): type of remote endpoint (either 'cpu' or 'gpu') - only used if local == False
    Returns:
        (str): URL of endpoint
    """
    if local:
        url = "http://127.0.0.1:5001"
        # print(f"Accessing local")
    else:
        if remote_type == 'cpu':
            url = "http://35.204.76.196:5001"
            # print(f"Accessing remote CPU")
        elif remote_type == 'v100':
            url = "http://35.204.119.9:5001"
            # print(f"Accessing remote GPU")
        elif remote_type == 't4':
            url = "http://35.204.91.169:5001"
        else:
            raise ValueError("remote_type must be either `cpu` or `gpu` for remote inference")

    return url


def set_local_model(model_type: str, num_threads: int = 1):
    global tokenizers, models

    # Set tokenizer
    if 'distil' in model_type:
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    elif 'bert' in model_type:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    elif 'electra' in model_type:
        tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')
    else:
        raise ValueError(f"Bad model type: {model_type}")

    # Get model path
    ## FIXME: change back to {model_type}-quant/{model_type}.onnx"
    model_path = f"models/{model_type}-opt.onnx"

    # Load model
    models[model_type] = create_model_for_provider(model_path, num_threads)
    tokenizers[model_type] = tokenizer

    print(f"Model set to {model_type} at {model_path}")


def set_model(model_type: str, local: bool = False, remote_type: str = None, onnx_quant: bool = False, num_threads: int = 1) -> None:
    global models

    if local:
        set_local_model(model_type, num_threads)
    else:
        url = get_url(local, remote_type)
        endpoint = remote_type

        response = requests.post(f"{url}/set_model", params={"model_type": model_type, "use_onnx_quant": onnx_quant, "num_threads": num_threads})
        if response.ok:
            print(f"Successfully set {endpoint} to {model_type}, ONNX quant: {onnx_quant}")
        else:
            raise RuntimeError(f"Failed to set {endpoint} to {model_type}, ONNX quant: {onnx_quant}")


def set_all_models():
    for model in model_types:
        set_model(model, local=True, num_threads=2)

    # set GPU model
    # set_model(model_types[0], local=False, remote_type='gpu', onnx_quant=False)
    return


def gather_sentences():
    global data

    # fetch the dataset
    train_set = load_dataset("glue", "sst2")['validation']
    sents = train_set[np.random.randint(0, train_set.num_rows, SENTENCES)]

    # map into proper format
    data = [{'sent': s,'label': l} for s,l in zip(sents['sentence'], sents['label'])]


def predict_model(model_type, input_string, label):
    # Perform inference
    start_time = time.time()

    inputs = tokenizers[model_type](input_string, return_tensors="np")
    # outputs = ort_session.run(["last_hidden_state"], dict(inputs))
    outputs = models[model_type].run([], dict(inputs))

    end_time = time.time()
    total_time = end_time - start_time

    # Get predictions
    logits = softmax(outputs[0])
    p_label = np.argmax(logits)
    prediction = int(p_label == label)

    input_size = inputs['input_ids'].shape[1]

    return {"confidence": round(logits[0, p_label], 3), "label": p_label,
            "score": prediction, "time": total_time, "input_size": input_size}


def update_aggregates(results, label):
    global model_metrics, sentence_table_log

    # get comparison points for speed up
    b_time = results['bert-base']['time']
    seq_len = results['bert-base']['input_size']

    # update model_metrics
    for model_type, metrics in results.items():
        speed_up = round(b_time / metrics['time'], 2)
        model_metrics[model_type]['speedup'].append(speed_up)
        model_metrics[model_type]['acc'].append(metrics['score'])

    # update sentence table log
    entry = [trial, seq_len, label]
    entry.extend([results[m]['label'] for m in model_types])

    # insert at top
    sentence_table_log.insert(0, entry)


@app.route('/process_sent/', methods=['POST'])
def process_sent():
    global trial

    # process request
    trial += 1
    form = eval(request.form.get('sent'))
    sent, label = form['sent'], form['label']

    # pass through models
    results = {}
    for model in model_types:
        results[model] = predict_model(model, sent, label)

    # update running tables
    update_aggregates(results, label)

    # prep global stats table
    global_table = [[mt, round(np.mean(val['acc']),3), round(np.mean(val['speedup']),2)] for mt, val in model_metrics.items()]

    # prep last experiment display
    last_exp = [[mt, val['label'], val['confidence'], round(model_metrics[mt]['speedup'][-1],2)] for mt,val in results.items()]
    last_data = {'sent':sent, 'label': label, 'seq_len': results['bert-base']['input_size']}

    return render_template('template.html', models=model_types, sentences=data,
                           log_table=sentence_table_log, global_table=global_table,
                           last_table=last_exp, last_data=last_data)


@app.route('/')
def main():
    global first_start

    if first_start:
        gather_sentences()
        set_all_models()
        initialize_tables()

        first_start = False

    return render_template('template.html', models=model_types, sentences=data,
                           log_table=sentence_table_log, last_data=None)


if __name__ == '__main__':
    app.run()

