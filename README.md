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