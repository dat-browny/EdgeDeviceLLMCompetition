# EdgeDeviceLLMCompetition

## Get data from Huggingface
Downloading data from Hugginface, we choose a proportion of a combined `c4` dataset then dump to useable data formats.

```bash
python data/get_data.py
```
This scripts will dump a part of `c4` to `data/c4/pruning_raw.jsonl` and `data/c4/cpt`.

## Competition Task 2: Pruning 
Ensure your environment using CUDA-Toolkit ver `11.8`.
```bash
nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```
Initialize requirement
```
conda create -n prune -y python=3.10
cd pruning & bash setup_env.sh
# Ignore the conflict version warining from pip.
```
Preprocess data into prunable type.
```
ROOT_PATH=$(pwd)

python pruning/convert_data.py \
    --target_dir $ROOT_PATH/data/c4/pruning/llama \
    --raw_data_dir $ROOT_PATH/data/c4/ \
    --tokenizer_name meta-llama/Llama-3.1-8B-Instruct \
    --dev_percent 0.01 \
    --max_seq_length 4096



python pruning/convert_data.py \
    --target_dir $ROOT_PATH/data/c4/pruning/qwen \
    --raw_data_dir $ROOT_PATH/data/c4/ \
    --tokenizer_name Qwen/Qwen2-7B-Instruct \
    --dev_percent 0.01 \
    --max_seq_length 4096



python pruning/convert_data.py \
    --target_dir $ROOT_PATH/data/c4/pruning/phi \
    --raw_data_dir $ROOT_PATH/data/c4/ \
    --tokenizer_name microsoft/phi-2 \
    --dev_percent 0.01 \
    --max_seq_length 4096
```
