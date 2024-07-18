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
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install 'llama-cpp-python[server]==0.2.64' 
```

Alternatively, we can skip using `llama-cpp-python` and use the [`llama-server`](https://github.com/ggerganov/llama.cpp/tree/master/examples/server) directly:

## Model Location
Models are currently stored in `/cluster/work/coss/LLMAgent/models`. 
We recommend creating a symbolic link to this folder to make accessing the models in the succeeding scripts easy:
[!NOTE] `/cluster/work` is optimized for IO operations, which influences how fast the model can be loaded in memory.

```
ln -s /cluster/work/coss/LLMAgent/models models
```

This will create a `models` folder in the root directory of this LLMagent repository.

Models can be downloaded from 

- 7B model: https://huggingface.co/TheBloke/Llama-2-7B-GGUF
- 13B model: https://huggingface.co/TheBloke/Llama-2-13B-GGUF
- 70B model: https://huggingface.co/TheBloke/Llama-2-70B-GGUF
- llama 3 model: `wget https://huggingface.co/PawanKrd/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/llama-3-70b-instruct.Q5_K_M.gguf`

## Deploying the model
To deploy the model, open a terminal window and run the following script (choose according to the model size):
```
# llama-2
python -m llama_cpp.server --model models/llama-2-70b-chat.Q8_0.gguf  --n_gpu_layers -1 --interrupt_requests f --n_batch 224 --chat_format llama-2 # 70b model

#llama-3
python -m llama_cpp.server --model models/llama-3-70b-instruct.Q5_K_M.gguf  --n_gpu_layers -1 --interrupt_requests f --n_batch 224 --chat_format llama-3 # 70b model

#llama-server
~/llama.cpp/llama-server --seed 42 -ngl 100 -b 224 --host 0.0.0.0 --port 8000 -chat-template llama2 -m models/llama-2-70b-chat.Q8_0.gguf
```

Just replace the `--model` argument with your desired model file.

## Running experiments
For running experiments in the Euler Jupyter cluster, it is necessary to first unset the `HTTP/S_PROXY` environment variables.
To do this, run
```
unset http_proxy https_proxy
```
in the terminal window that you will use to execute the script. 
Alternatively, one can remove these environment variables by modifying `sys.path` within the python script.
