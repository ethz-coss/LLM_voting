# Running on EULER

Please follow this guide for running experiments on the ETH supercomputing cluster Euler.


## Requesting a GPU instance
We recommend using the Euler jupyterlab interface (https://jupyter.euler.hpc.ethz.ch) to launch a server with GPUs.
The recommended arguments are:

- number of processors: 8
- RAM: 64GB
- number of GPUs: 2
- GPU model (under additional options): rtx_4090 or rtx_3090


## Setup
This repository relies on the [lamma-cpp-python](https://github.com/abetlen/llama-cpp-python) package for exposing a [OpenAI-like API](https://platform.openai.com/) for the LLaMa2 model.
We first need to install the package with GPU acceleration using the following command:

```
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install 'llama-cpp-python[server]==0.2.17' 
```


## Model Location
Models are currently stored in `/cluster/project/gess/coss/projects/LLMAgent/models`.
We recommend creatig a symbolic link to this folder to make accessing the models in the succeeding scripts easy:

```
ln -s /cluster/project/gess/coss/projects/LLMAgent/models models
```

This will create a `models` folder in the root directory of this LLMagent repository.

Models can be downloaded from 

- 7B model: https://huggingface.co/TheBloke/Llama-2-7B-GGUF
- 13B model: https://huggingface.co/TheBloke/Llama-2-13B-GGUF
- 70B model: https://huggingface.co/TheBloke/Llama-2-70B-GGUF


## Deploying the model
To deploy the model, open a terminal window and run the following script (choose according to the model size):

```
python -m llama_cpp.server --model models/llama-2-70b-chat.Q4_K_M.gguf  --n_gpu_layers 83 --interrupt_requests f --n_batch 224 # 70b model
```

Just replace the `--model` argument with your desired model file.

## Running experiments
For running experiements in the Euler Jupyter cluster, it is necessary to first unset the `HTTP/S_PROXY` environment variables.
To do this, run
```
unset http_proxy https_proxy
```
in the terminal window that you will use to execute the script. 
Alternatively, one can remove these environment variables by modifying `sys.path` within the python script.