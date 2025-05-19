

```
conda create -n dvs_prun python=3.9
conda activate dvs_prun
conda install -c conda-forge libstdcxx-ng

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install omegaconf opencv-python matplotlib psutil wandb lightning numba pybind11 tqdm pandas

python setup.py build_ext --inplace
```