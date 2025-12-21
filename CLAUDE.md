# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TRELLIS.2** is a 4B-parameter image-to-3D generative model developed by Microsoft Research. It generates high-fidelity 3D assets with arbitrary topology (open surfaces, non-manifold geometry, internal structures) and full PBR materials (base color, metallic, roughness, alpha) using a novel "O-Voxel" sparse voxel representation.

## Development Commands

### Environment Setup
```bash
# Full installation (creates conda environment 'trellis2')
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

# Modular installation (if issues occur)
. ./setup.sh --new-env      # Create conda environment
. ./setup.sh --basic        # Install basic dependencies
. ./setup.sh --flash-attn   # Install flash-attention (CUDA only)
. ./setup.sh --nvdiffrast   # Install differentiable rasterizer
. ./setup.sh --nvdiffrec    # Install PBR renderer
. ./setup.sh --cumesh       # Install CUDA mesh utilities
. ./setup.sh --o-voxel      # Install O-Voxel library
. ./setup.sh --flexgemm     # Install sparse convolution
```

**Important Notes:**
- Requires NVIDIA GPU with ≥24GB VRAM (tested on A100, H100)
- CUDA Toolkit 12.4 recommended (11.8+ supported)
- Linux only (not tested on Windows/macOS)
- If multiple CUDA versions installed: `export CUDA_HOME=/usr/local/cuda-12.4`
- For GPUs without flash-attn (e.g., V100): Install xformers and set `export ATTN_BACKEND=xformers`

### Running the Code

```bash
# Minimal example (generates sample.mp4 and sample.glb)
python example.py

# Web demo (Gradio interface)
python app.py
```

**Required Environment Variables:**
```bash
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # GPU memory optimization
export ATTN_BACKEND=xformers  # Optional: for GPUs without flash-attn
```

### Testing
No formal test framework is currently configured. Use `example.py` and `app.py` for validation.

## Code Architecture

### High-Level Pipeline Flow
```
Input Image
  ↓ (preprocess, remove background)
  ↓ (extract image features via CLIP/T5)
  ↓ Stage 1: Sample sparse structure (32³ or 64³ binary occupancy)
  ↓ Stage 2: Sample shape latent (cascade: 512³ → 1024³ → 1536³)
  ↓ Stage 3: Sample texture latent (PBR attributes)
  ↓ (decode latents to mesh + voxel attributes)
  → MeshWithVoxel (vertices, faces, PBR materials)
```

### Core Components

**1. Pipeline Layer** (`/trellis2/pipelines/`)
- `Trellis2ImageTo3DPipeline`: Main orchestrator for image→3D generation
- `Trellis2ImageTo3DCascadePipeline`: Supports multi-resolution cascade (512→1024→1536)
- `Trellis2ImageToTexturePipeline`: Texture-only generation (coming soon)

**2. Generative Models** (`/trellis2/models/`)
- `SparseStructureFlowModel`: Dense DiT for coarse occupancy structure (32³/64³)
- `SLatFlowModel`: Sparse transformer for shape/texture latents (operates on SparseTensor)
  - Separate instances: `shape_slat_flow_model_512`, `tex_slat_flow_model`
  - Supports cascade upsampling with token limits (max 49,152 tokens)
- `ElasticSLatFlowModel`: Extended version with elastic memory management

**3. Sparse Representations** (`/trellis2/modules/sparse/`)
- `SparseTensor`: Custom sparse data structure
  - Coordinates: (batch_idx, x, y, z) + feature vectors
  - Supports variable-length batching, concatenation, slicing
  - Operations: `sparse_cat`, `replace`, device movement, batch indexing
- `VarLenTensor`: Parent class for variable-length sequential data
- Sparse operations: `SparseConv3d`, `SparseLinear`, `SparseUpsample`, `SparseDownsample`

**4. Transformer Architecture** (`/trellis2/modules/`)
- `ModulatedSparseTransformerCrossBlock`: Core building block
  - Sparse self-attention (only on occupied voxels)
  - Cross-attention to image conditioning
  - Per-token feedforward MLP
  - RoPE position encoding
- Attention backends: flash-attn (default), xformers (fallback for older GPUs)

**5. VAE Decoders** (`/trellis2/models/sc_vaes/`)
- `SparseUnetVaeDecoder`: Converts latents → mesh geometry or PBR attributes
  - Shape decoder: SparseTensor → vertices, faces, voxel structure
  - Texture decoder: SparseTensor → 6-channel (base_color(3), metallic, roughness, alpha)
  - Sparse upsampling stages with token management

**6. Samplers** (`/trellis2/pipelines/samplers/`)
- `FlowEulerSampler`: Flow matching with Euler integration
  - Forward: `x_t = (1-t) * x_0 + σ(t) * noise`
  - Reverse: `x_{t-1} = x_t - dt * v_pred`
- `FlowEulerCfgSampler`: Adds classifier-free guidance
- `FlowEulerGuidanceIntervalSampler`: Guidance only in specific timestep ranges

**7. O-Voxel System** (`/o-voxel/`)
- Sparse voxel representation supporting arbitrary topology
- Python bindings with CUDA/ROCm extensions
- Key modules:
  - `convert/flexible_dual_grid.py`: Mesh ↔ voxel conversion
  - `postprocess.py`: Decimation, remeshing, UV-unwrapping, GLB export
  - `io/`: Read/write .vxz, .npz, .ply formats
  - `rasterize.py`: Attribute rasterization to voxel grid

**8. Renderers** (`/trellis2/renderers/`)
- `PbrMeshRenderer`: Real-time PBR rendering with environment maps
- `EnvMap`: HDRI environment lighting
- Uses nvdiffrast (CUDA-accelerated differentiable rasterization)

### Key Architectural Patterns

**Cascade Generation Strategy:**
- Low-res generation (512³) → upsampling → high-res refinement (1024³/1536³)
- Adaptive token limiting: Reduces resolution to 1024³ if token count exceeds 49,152
- Separate flow models for each resolution

**Flow Matching (vs Diffusion):**
- Uses v-prediction (velocity field) parameterization
- Linear interpolation: `x_t = (1-t) * x_0 + t * noise`
- Sigma scaling: `σ(t) = σ_min + (1-σ_min) * t`
- More efficient than traditional diffusion

**Sparse Computing Throughout:**
- Avoids dense 3D grids (memory intensive)
- Coordinate-based sparse tensors with FlexGEMM (Triton-based sparse conv)
- Benefits: Handles arbitrary topology naturally, scales to high resolutions

**Low-VRAM Mode:**
- Models loaded/unloaded from GPU on-demand
- Enables 4B-parameter model on 24GB GPUs
- Controlled via pipeline configuration

**Three-Stage Independence:**
- Each stage (structure → shape → texture) is a separate flow model
- Progressive refinement: Each builds on previous stage
- Classifier-free guidance at each stage for quality control

## Critical File Locations

### Model Implementations
- `/trellis2/models/sparse_structure_flow.py` - Structure generation model
- `/trellis2/models/structured_latent_flow.py` - Shape/texture latent models
- `/trellis2/models/sc_vaes/slat_unet_vae.py` - VAE decoder architecture

### Pipeline Orchestration
- `/trellis2/pipelines/trellis2_image_to_3d.py` - Main image→3D pipeline (direct)
- `/trellis2/pipelines/trellis2_image_to_3d_cascade.py` - Cascade pipeline (multi-res)
- `/trellis2/pipelines/base.py` - Base Pipeline class with config loading

### Sparse Tensor System
- `/trellis2/modules/sparse/basic.py` - SparseTensor, VarLenTensor core
- `/trellis2/modules/sparse/conv/sparse_conv.py` - Sparse convolutions
- `/trellis2/modules/sparse/transformer/sparse_transformer.py` - Sparse attention

### O-Voxel Integration
- `/o-voxel/o_voxel/convert/flexible_dual_grid.py` - Mesh ↔ voxel conversion
- `/o-voxel/o_voxel/postprocess.py` - GLB export, remeshing, decimation
- `/o-voxel/setup.py` - CUDA/ROCm extension build configuration

### Entry Points
- `/example.py` - Minimal usage example (47 lines)
- `/app.py` - Gradio web demo (645 lines)
- `/setup.sh` - Installation script

## Model Loading

Pretrained models load automatically from Hugging Face:
```python
from trellis2.pipelines import Trellis2ImageTo3DPipeline

# Loads microsoft/TRELLIS.2-4B from Hugging Face Hub
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()
```

Model components (defined in pipeline config):
- `sparse_structure_flow_model` - Coarse structure generation
- `shape_slat_flow_model_512` - 512³ shape latent
- `shape_slat_flow_model_1024` - 1024³ shape latent (cascade only)
- `tex_slat_flow_model` - Texture latent
- `slat_decoder_sh` / `slat_decoder_tex` - VAE decoders

## Special Considerations

**Memory Management:**
- Large models (4B parameters) require careful GPU memory handling
- Use `expandable_segments:True` for dynamic memory allocation
- Token limits prevent OOM: max 49,152 tokens per batch
- Low-VRAM mode available via config (loads models on-demand)

**Topology Handling:**
- O-Voxel representation is "field-free" (no signed distance functions)
- Supports open surfaces, non-manifold geometry, internal structures
- No iso-surface extraction required (unlike NeRF/SDF methods)
- Direct mesh extraction via dual grid construction

**PBR Materials:**
- 6-channel output: base_color (RGB), metallic, roughness, alpha
- GLB export defaults to OPAQUE mode (alpha preserved but not active)
- To enable transparency: Manually connect alpha channel in 3D software

**Hardware Compatibility:**
- CUDA: Full support (flash-attn, nvdiffrast, nvdiffrec)
- ROCm (AMD): Partial support via HIP (no flash-attn/nvdiffrast)
- CPU: Not supported for inference (GPU required)

**Future Releases (per README):**
- Shape-conditioned texture generation (before 12/24/2025)
- Training code (before 12/31/2025)

## External Resources

- **Paper:** https://arxiv.org/abs/2512.14692
- **Hugging Face Model:** https://huggingface.co/microsoft/TRELLIS.2-4B
- **Hugging Face Demo:** https://huggingface.co/spaces/microsoft/TRELLIS.2
- **Project Website:** https://microsoft.github.io/TRELLIS.2
- **Related Packages:**
  - FlexGEMM: https://github.com/JeffreyXiang/FlexGEMM (Triton sparse conv)
  - CuMesh: https://github.com/JeffreyXiang/CuMesh (CUDA mesh utilities)
