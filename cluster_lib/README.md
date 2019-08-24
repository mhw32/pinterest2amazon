## Environment Setup
```
conda create -n python3 anaconda
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
pip install tqdm
pip install dotmap
conda install faiss-gpu cudatoolkit=9.0 -c pytorch
pip install torchtext
```

## Runtime Setup
```
conda activate python3
source init_env.sh
```

## Train Clustering Model
```
python scripts/instance.py ./config/instance.json
python scripts/localagg.py ./config/localagg.json
```
