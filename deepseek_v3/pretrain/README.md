# DeepSeek V3 Pre-training

This directory contains performance recipes for DeepSeek V3 pre-training workloads using different frameworks.

## Available Frameworks

### Megatron-Bridge

- **Path**: [megatron_bridge/](megatron_bridge/README.md)
- **Description**: Training using NeMo Megatron-Bridge framework
- **Model Size**: 671B parameters
- **Supported GPUs**: VR NVL72, GB300, GB200, B300, B200, H100
- **Precision**: NVFP4, FP8, BF16

### Bare Metal (Vera Rubin Alpha)

- **Path**: [baremetal/](baremetal/README.md)
- **Description**: Experimental alpha workflow for running DeepSeek V3 pre-training with `torchrun` directly on bare-metal Vera Rubin trays
- **Model Size**: 671B parameters (plus smoke and proxy modes)
- **Supported GPUs**: VR NVL72
- **Precision**: NVFP4, FP8, BF16

## Quick Start

Please refer to the framework-specific README files for detailed setup and running instructions:

- [Megatron-Bridge README](megatron_bridge/README.md)
- [Bare-Metal Vera Rubin README](baremetal/README.md)

For complete installation instructions, see the [main repository README](../../README.md).
