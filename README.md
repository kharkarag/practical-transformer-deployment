# practical-transformer-deployment
Final project for COMS E6998 Practical Deep Learning Systems Performance

## Code Layout

### Training and Preparation
`run_glue.py`: part of the Huggingface docs for training/evaluating models on the GLUE benchmark. We use it to finetune each of the models.

`finetune.sh`: convenience script for finetuning the models. Runs `run_glue.py` with specific parameters.

`setup.sh`: covenience script to set up the environment for experiments. Clones the repo, creates a conda environment using `environnment.yml`, and downloads/converts all the models.

### Experiments
`client.py`: entry point for experiments. Sends inference requests to the servers hosting models. Can optionally a full experiment: trials for all combinations of parameters.  
Example usage: `python client.py --run_full --num_trials 100`

`inference.py`: Flask app for hosting models. API includes `set_model` for loading a model from disk and `inference` for executing loaded model on inputs.  
Usage: `export FLASK_APP=inference && flask run --host=0.0.0.0 --port=5001`


### Utilities
`convert_graph_to_onnx.py`: this script is part of the ONNX runtime docs. It converts PyTorch models to ONNX, and can optionally optimize and quantize them.

`convert.sh`: convenience script to download and convert all 4 PyTorch models to ONNX.

`quantize.sh`: convenience script to download, convert, and quantize all 4 PyTorch models to ONNX.


## Demo
TODO


## Experiments and Results

**Case Study:** You are working on a sentiment model at your company. This task is latency conscious and you have access to limited resources (budgeted deployment). You want to assert to your team that your developed model is predictively powerful, yet fast enough to meet latency requirements.

Deployment Strategies:
* Model on device, single CPU core
* Model on server, single or multiple CPU cores
* Model on server, single GPU

Architecture Sizes
* BERT-base
* BERT-tiny
* DistilBERT
* ELECTRA-small

Hardware Versions
* Local: 1-core Skylake CPU
* Server (CPU): 8-core Skylake CPU
* Server (GPU): 1 x V100 GPU


![Model Summary](img/model_summary.png "Model Summary")


![Time vs. Input Size](img/time_vs_input.png "Time vs. Input Size")


![Speedup vs. Batch Size](img/speedup.png "Speedup vs. Batch Size")


![All vs. All](img/all_vs_all.png "All vs. All")


![Snapshot of Largest Input Size](img/largest_input_snapshot.png "Snapshot of Largest Input Size")


![ONNX Optimization](img/onnx_opt.png "ONNX Optimization")
