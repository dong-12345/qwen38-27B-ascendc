---
license: apache-2.0
tags:
- vision
- moondream
- mlx
- int4
---

# MD3 Preview - Int4 Quantized (MLX)

Pre-quantized version of Moondream 3 Preview for MLX inference.

## Quantization Details

- **MoE Experts**: int4 affine quantization (bits=4, group_size=64)
- **Other weights**: bf16 (unchanged)
- **Memory savings**: ~60% reduction in MoE weight memory

## Source

Quantized from [moondream/moondream3-preview](https://huggingface.co/moondream/moondream3-preview)
