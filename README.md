# Project: ICL Cache

Environment:
```bash
conda create -n sysproj python=3.10
conda activate sysproj
cd 24MLSYS-prompt-cache
pip install -r requirements.txt
```

Run benchmark check:
```bash
conda activate sysproj
cd 24MLSYS-prompt-cache
python -m benchmark.longbench
```

Run evaluation:
```bash
conda activate sysproj
cd 24MLSYS-prompt-cache
python eval.py --dataset squad_v2
python eval.py --dataset icl_symbol
python eval.py --enable_cache=True --dataset icl_symbol

python eval.py --dataset icl_riddlesense
python eval.py --enable_cache=True --dataset icl_riddlesense
```

Run demo:
```bash
conda activate sysproj
cd 24MLSYS-prompt-cache

python demo_math.py --cuda_device 2 > outputs/math_no_cache_codellama.log # could not run with 12GB GPU due to cache size

python demo_riddle.py --enable_cache True > outputs/riddle_with_cache_llama2.log # use 8bit quantization on 12GB GPU
python demo_riddle.py --cuda_device 1 > outputs/riddle_no_cache_llama2.log # use 8bit quantization on 12GB GPU
python demo_riddle.py --cuda_device 2 --result_file_name riddle_with_cache_llama2_newprompt --enable_cache True > outputs/riddle_with_cache_llama2_newprompt.log
python demo_riddle.py --cuda_device 3 --result_file_name riddle_no_cache_llama2_newprompt > outputs/riddle_no_cache_llama2_newprompt.log

python demo_csqa.py --cuda_device 4 --result_file_name csqa_no_cache_llama2 > outputs/csqa_no_cache_llama2.log
python demo_csqa.py --cuda_device 5 --result_file_name csqa_with_cache_llama2 --enable_cache True > outputs/csqa_with_cache_llama2.log

python demo_sst2.py --cuda_device 6 --result_file_name sst2_no_cache_llama2 > outputs/sst2_no_cache_llama2.log
python demo_sst2.py --cuda_device 7 --result_file_name sst2_with_cache_llama2 --enable_cache True > outputs/sst2_with_cache_llama2.log

python demo_wmt.py --cuda_device 0 --result_file_name wmt_no_cache_llama2 > outputs/wmt_no_cache_llama2.log
python demo_wmt.py --cuda_device 1 --result_file_name wmt_with_cache_llama2 --enable_cache True > outputs/wmt_with_cache_llama2.log
```