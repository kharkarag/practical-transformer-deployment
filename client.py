import requests
import json
import time
import argparse


def inference_request(url, batch_size):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = [
        "This is a test string for the various transformer models to evaluate."
    ] * batch_size
    return requests.post(url, json=data, headers=headers)



def get_url(local, remote_type=None):
    if local:
        url = "http://127.0.0.1:5001"
        print(f"Accessing local")
    else:
        if remote_type == 'cpu':
            url = "http://35.204.76.196:5001"
            print(f"Accessing remote CPU")
        elif remote_type == 'gpu':
            url = "http://35.204.119.9:5001"
            print(f"Accessing remote GPU")
        else:
            raise ArgumentError("remote_type must be either `cpu` or `gpu` for remote inference")
        
    return url

def set_model(model_type, local=False, remote_type=None, onnx_opt=False):
    url = get_url(local, remote_type)
    response = requests.post(f"{url}/set_model", params={"model_type": model_type, "use_onnx_optim": onnx_opt})
    
    if response.ok:
        endpoint = 'local' if local else remote_type
        print(f"Successfully set {endpoint} to {model_type}, ONNX opt: {onnx_opt}")
    else:
        raise RuntimeError(f"Failed to set {endpoint} to {model_type}, ONNX opt: {onnx_opt}")
    
def run_trials(model_type, batch_size=32, local=False, remote_type=None, onnx_opt=False):
    num_trials = 5
    
    url = get_url(local, remote_type)
    print(f"Using batch size: {batch_size}")
        
    times = []
    for i in range(num_trials):    
        start_time = time.time()
        response = inference_request(f"{url}/inference", batch_size)
        end_time = time.time()
        output = json.loads(response.content)
        
        if local:
            t = output['time']
        else:
            t = end_time - start_time
        times.append(t)
        
    print(f"Average time: {sum(times)/num_trials:.3f} s")
    return times


def run_all_endpoints(model_type, batch_size, onnx_opt):
    local_cpu_times = run_trials(model_type, batch_size=batch_size, local=True, onnx_opt=onnx_opt)
    remote_cpu_times = run_trials(model_type, batch_size=batch_size, local=False, remote_type='cpu', onnx_opt=onnx_opt)
    remote_gpu_times = run_trials(model_type, batch_size=batch_size, local=False, remote_type='gpu', onnx_opt=onnx_opt)
    
    results = dict()
    results['local_cpu'] = local_cpu_times
    results['remote_cpu'] = remote_cpu_times
    results['remote_gpu'] = remote_gpu_times
    return results


def set_all_models(model_type, onnx_opt):
    set_model(model_type, local=True, onnx_opt=onnx_opt)
    set_model(model_type, local=False, remote_type='cpu', onnx_opt=onnx_opt)
    set_model(model_type, local=False, remote_type='gpu', onnx_opt=onnx_opt)
    

model_types = ['bert-base', 'bert-tiny', 'distilbert', 'electra-small']
onnx_opts = [False, True]
batch_sizes = [2**i for i in range(7)]
    
def run_full_project():
    master_results = dict()
    
    for model_type in model_types:
        onnx_opt_results = dict()
        for onnx_opt in onnx_opts:
            set_all_models(model_type, onnx_opt)
            batch_size_results = dict()
            for batch_size in batch_sizes:
                print(f"Running {model_type}, optim {onnx_opt}, batch_size {batch_size}")
                times = run_all_endpoints(model_type, batch_size, onnx_opt)
                batch_size_results[batch_size] = times
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
    parser.add_argument('--onnx_opt', action='store_true', default=False)
    parser.add_argument('--run_full', action='store_true', default=False)
    args = parser.parse_args()
    
    if args.run_full:
        run_full_project()
    else:
        response = set_model(args.model_type, local=args.local, remote_type=args.remote_type, onnx_opt=args.onnx_opt)
        times = run_trials(args.model_type, batch_size=args.batch_size, local=args.local, remote_type=args.remote_type, onnx_opt=args.onnx_opt)
