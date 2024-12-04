# Project: ICL Cache

## Run PromptCache
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
python demo_riddle.py --enable_cache True > outputs/riddle_with_cache_llama2.log # use 8bit quantization on 12GB GPU
python demo_riddle.py --cuda_device 1 > outputs/riddle_no_cache_llama2.log # use 8bit quantization on 12GB GPU
python demo_math.py --cuda_device 2 > outputs/math_no_cache_codellama.log # could not run with 12GB GPU due to cache size
```