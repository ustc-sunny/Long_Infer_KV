# Speedup Evaluation Test
This directory contains source codes for evaluating the speedup. You can reproduce inference latency results in the paper. Some of the codes are referenced from FlexGen (ICML'23) GitHub repository.
- Getting Started (10 minutes)

## Getting Started (10 minutes)
```sh
sh install.sh
export CUDA_HOME=/path/to/cuda
```
```sh
#install.sh
pip install -e longinfer
pip install -e flexgen
```
For a "Hello world"-sized example, please run the following command (10 minutes):
```
python -m flexgen.flex_opt --model huggingface/opt-6.7b --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 4 --num-gpu-batches 1 --prompt-len 384 --gen-len 128 --warmup-input-path flexgen/pg19_firstbook.txt --test-input-path flexgen/pg19_firstbook.txt --importance_ratio 0.2
```
## About the test
You can change the parameters of above instruct to test the performances.

