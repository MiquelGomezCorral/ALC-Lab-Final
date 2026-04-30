# -------------------------------------------------
# 0. ACTIVATE ENV
# -------------------------------------------------
conda activate ALC_env_2

# -------------------------------------------------
# 1. REMOVE ALL CURRENT PACKAGES FROM ENV
# -------------------------------------------------
uv pip freeze > installed.txt
uv pip uninstall -r installed.txt
rm installed.txt

# -------------------------------------------------
# 2. UPGRADE BUILD TOOLS
# -------------------------------------------------
uv pip install --upgrade pip setuptools wheel

# -------------------------------------------------
# 3. INSTALL PYTORCH CUDA 12.4
# (change to cpu wheels if no GPU)
# -------------------------------------------------
uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

# -------------------------------------------------
# 4. INSTALL CLEAN STACK
# -------------------------------------------------
uv pip install \
transformers==4.57.1 \
accelerate==1.10.1 \
huggingface_hub==0.34.4 \
datasets==4.0.0 \
tokenizers==0.22.0 \
safetensors==0.6.2 \
sentencepiece==0.2.0 \
numpy==2.2.6 \
pillow==11.3.0 \
tqdm==4.67.1 \
pyarrow==21.0.0 \
pandas==2.3.2 \
decord==0.6.0 \
einops==0.8.1 \
scipy==1.16.1

# -------------------------------------------------
# 5. OPTIONAL FLASH-ATTN
# (Linux + NVIDIA only; skip if it errors)
# -------------------------------------------------
uv pip install flash-attn --no-build-isolation

# -------------------------------------------------
# 6. VERIFY
# -------------------------------------------------
python -c "import torch, transformers, datasets; print(torch.__version__); print(transformers.__version__); print(datasets.__version__)"