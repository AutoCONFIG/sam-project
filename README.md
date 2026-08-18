# SAM 3 Project — Unified CLI Framework

Unified entry point for SAM 3 (Segment Anything Model 3) inference and training,
with YAML configuration mounting and CLI override support.

## Architecture

```
sam-project/
├── sam.py                # Unified entry: reads YAML `mode` → dispatches to commands/
├── commands/             # predict.py, train.py
├── core/                 # engine.py (SAM3 predictor wrapper), visualization.py
├── utils/                # config.py, constants.py
├── configs/              # YAML configs for predict/train
├── sam3/                 # git submodule (SAM3 backend, fork: AutoCONFIG/sam3)
└── runs/                 # Output directory (gitignored)
```

The SAM3 backend is a git submodule pointing to [AutoCONFIG/sam3](https://github.com/AutoCONFIG/sam3)
(fork of [facebookresearch/sam3](https://github.com/facebookresearch/sam3)).

## Installation

### 1. Clone with submodules

```bash
git clone --recursive git@github.com:AutoCONFIG/sam-project.git
cd sam-project
```

If already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### 2. Install PyTorch (CUDA 12.8+ for Blackwell GPUs)

```bash
# For RTX 50-series (Blackwell, sm_120) — requires CUDA 12.8+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# For older GPUs (Ampere/Ada, sm_80+)
pip install torch torchvision
```

### 3. Install SAM 3 backend and dependencies

```bash
cd sam3
pip install -e .
pip install -e ".[train]"   # if training
cd ..

# Frontend dependencies
pip install pyyaml opencv-python numpy
```

### 4. Download model weights

Download `sam3.1_multiplex.pt` from [ModelScope](https://modelscope.cn/models/facebook/SAM3.1)
or [Hugging Face](https://huggingface.co/facebook/sam3.1).

## Usage

### Inference (video with text prompts)

```bash
# Edit configs/predict/video_text.yaml to set checkpoint path, input video, and text prompt
python sam.py configs/predict/video_text.yaml

# CLI overrides (any YAML value can be overridden)
python sam.py configs/predict/video_text.yaml --text "dog" --image-size 672
python sam.py configs/predict/video_text.yaml -i /path/to/video.mp4 -o runs/predict/custom
```

### Training

```bash
# Create a train config (see configs/train/README.md)
python sam.py configs/train/<your_config>.yaml

# CLI overrides
python sam.py configs/train/<your_config>.yaml --num-gpus 2
```

### Direct CLI (without YAML)

```bash
python -m commands.predict \
    --checkpoint /path/to/sam3.1_multiplex.pt \
    --input video.mp4 \
    --output runs/predict/test \
    --text "person" \
    --image-size 672
```

## Configuration

All config follows the YAML + CLI override pattern:

1. Load YAML config file
2. Parse CLI arguments
3. Deep-merge: CLI values override YAML (only for explicitly specified args)
4. Dispatch to the appropriate command

See `configs/predict/video_text.yaml` for a complete example.

## Key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `model.image_size` | 1008 | Must be multiple of 336. Use 672 for 16GB GPUs. |
| `model.use_fa3` | false | Flash Attention 3 (requires flash-attn). |
| `model.use_rope_real` | true | Real-valued RoPE (avoids complex64 buffers). |
| `prompt.text` | (required) | Open-vocabulary text prompt, e.g. "person". |
| `io.save_vis` | true | Save visualization overlays. |
| `io.save_masks` | true | Save mask npz files (label_map + metadata). |
