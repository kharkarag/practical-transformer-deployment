import json
import time
import requests
import argparse
import numpy as np


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
    data = [
        "String which will be used to evaluate model speed." * seq_multiplier
    ] * batch_size

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
        elif remote_type == 'gpu':
            url = "http://35.204.119.9:5001"
            # print(f"Accessing remote GPU")
        else:
            raise ValueError("remote_type must be either `cpu` or `gpu` for remote inference")

    return url


def set_model(model_type: str, local: bool = False, remote_type: str = None, onnx_quant: bool = False) -> None:
    """
    Sends a request to set the model type of the specifies endpoint and verifies the response.
    Args:
        model_type (str): model to load on the endpoint. Must be in ['bert-base', 'bert-tiny', 'distilbert', 'electra-small']
        local (bool): whether to access the local endpoint or a remote one
        remote_type (str): type of remote endpoint (either 'cpu' or 'gpu') - only used if local == False
        onnx_quant (bool): whether to use ONNX-quantized model
    """
    url = get_url(local, remote_type)
    endpoint = 'local' if local else remote_type

    response = requests.post(f"{url}/set_model", params={"model_type": model_type, "use_onnx_quant": onnx_quant})
    if response.ok:
        print(f"Successfully set {endpoint} to {model_type}, ONNX quant: {onnx_quant}")
    else:
        raise RuntimeError(f"Failed to set {endpoint} to {model_type}, ONNX quant: {onnx_quant}")


def run_trials(batch_size: int = 32, seq_multi: int = 1 , local: bool = False,
               remote_type: str = None, num_trials: int = 500) -> list:
    """
    Runs several trials of model inferences and collects timing data for each trial.
    Args:
        batch_size (int): number of dummy sentences in the batch
        local (bool): whether to access the local endpoint or a remote one
        remote_type (str): type of remote endpoint (either 'cpu' or 'gpu') - only used if local == False
    Returns:
        (list): timing data metrics for each trial
    """
    url = get_url(local, remote_type)
    print(f"Using batch size: {batch_size}, seq_len: {seq_multi * 2 + 1}")

    times = []
    for _ in range(num_trials):
        response, run_time = inference_request(f"{url}/inference", batch_size, seq_multi)
        output = json.loads(response.content)

        times.append({'local_time': output['time'], 'total_time': run_time})

    print(f"Average total time: {sum([t['total_time'] for t in times])/num_trials:.3f} s")
    return times


def run_all_endpoints(batch_size: int, seq_multi: int) -> dict:
    """
    Sends an inference request to all of the endpoints.
    Args:
        batch_size (int): number of dummy sentences in the batch
    Returns:
        (dict): timing metrics for each endpoint
    """
    local_cpu_times = run_trials(batch_size=batch_size, seq_multi=seq_multi, local=True)
    remote_cpu_times = run_trials(batch_size=batch_size, seq_multi=seq_multi, local=False, remote_type='cpu')
    remote_gpu_times = run_trials(batch_size=batch_size, seq_multi=seq_multi, local=False, remote_type='gpu')

    results = dict()
    results['local_cpu'] = local_cpu_times
    results['remote_cpu'] = remote_cpu_times
    results['remote_gpu'] = remote_gpu_times
    return results


def set_all_models(model_type: str, onnx_quant: bool) -> None:
    """
    Sets the model for all endpoints.
    Args:
        model_type (str): model to load on the endpoint. Must be in ['bert-base', 'bert-tiny', 'distilbert', 'electra-small']
        onnx_quant (bool): whether to use ONNX-quantized model
    """
    set_model(model_type, local=True, onnx_quant=onnx_quant)
    set_model(model_type, local=False, remote_type='cpu', onnx_quant=onnx_quant)
    set_model(model_type, local=False, remote_type='gpu', onnx_quant=onnx_quant)


def run_full_project(num_trials: int = 5):
    """
    Runs the entire project.
    Performs inference for all model types (with and without ONNX optim) for all batch sizes across all endpoints.
    """
    model_types = ['bert-base', 'bert-tiny', 'distilbert', 'electra-small']
    onnx_quants = [False, True]
    batch_sizes = [2**i for i in range(7)]
    seq_lens = [x for x in range(25)]
    master_results = dict()

    for model_type in model_types:
        onnx_opt_results = dict()

        for onnx_quant in onnx_opts:
            set_all_models(model_type, onnx_quant)
            batch_size_results = dict()

            for batch_size in batch_sizes:
                s_decay = 10 if batch_size >= 64 else len(seq_lens)

                for seq_m in seq_lens[:-s_decay]:
                    print(f"Running {model_type}, optim {onnx_opt}, batch_size {batch_size}, seq_len {seq_m * 10 + 2}")
                    times = run_all_endpoints(batch_size, seq_m)
                    time_metrics = {k:(np.mean(v), np.std(v)) for k, v in times.items()}
                    batch_size_results[(batch_size, seq_m)] = time_metrics

            onnx_opt_results[onnx_opt] = batch_size_results
        master_results[model_type] = onnx_opt_results

    with open('master_results.json', 'w+') as f:
        json.dump(master_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('--remote_type', type=str, default=None)
    parser.add_argument('--onnx_quant', action='store_true', default=False)
    parser.add_argument('--run_full', action='store_true', default=False)
    parser.add_argument('--num_trials', type=int, default=5)
    args = parser.parse_args()

    if args.run_full:
        run_full_project(args.num_trials)
    else:
        response = set_model(args.model_type, local=args.local, remote_type=args.remote_type, onnx_quant=args.onnx_quant)
        times = run_trials(batch_size=args.batch_size, local=args.local, remote_type=args.remote_type)
