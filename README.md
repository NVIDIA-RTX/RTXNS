# RTX Neural Shading

RTX Neural Shading (RTXNS), also known as RTX Neural Shaders, is a starting point for developers interested in bringing machine learning (ML) to graphics applications on Windows or Linux. It includes examples that show how to train neural networks and use the resulting models for inference alongside conventional graphics rendering.

RTXNS uses the [Slang](https://shader-slang.com) shading language and either the DirectX Preview Agility SDK or the Vulkan Cooperative Vector extension to access GPU ML acceleration.

The examples build on one another, progressing from simple inference to training a neural network to represent a shader or texture. The SDK also includes helper functions for building custom neural networks.

The SlangPy samples demonstrate rapid neural-network development in Python and how to integrate the resulting implementation into RTXNS for inference.

When exploring RTXNS, it is assumed that the reader is already familiar with ML and neural networks.

## Requirements

### General

[CMake 3.24 or later][CMake] **|** [Slang 2026.10*](https://shader-slang.com/tools/)

### Windows

[Visual Studio 2022 or later][VisualStudio]

Both x64 and Arm64 (Windows on Arm) targets are supported. Architecture-specific dependencies such as NVAPI, the Agility SDK, DXC, and Slang are selected for the target or build host as appropriate. `CMakePresets.json` provides Ninja-based Debug and Release presets for native and cross builds. Ninja is required when using these presets, but it is not required by the SDK: custom CMake builds can continue to use Visual Studio or another supported generator. From a command line, run a preset in the Visual Studio Developer Command Prompt for its target architecture. Visual Studio and VS Code can select the same presets directly. See the [Quick Start Guide](docs/QuickStart.md) for commands and cross-compilation details.

### Linux

[Ninja][Ninja]

### DirectX (Windows only)

[DirectX Preview Agility SDK 1.721.3-preview*](https://www.nuget.org/packages/Microsoft.Direct3D.D3D12/1.721.3-preview) **|** [Microsoft DXC v1.10.2605.24*](https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.10.2605.24) **|** [NVIDIA public R615 driver or newer](https://www.nvidia.com/en-us/drivers/)

### Vulkan (Windows and Linux)

GPU must support the Vulkan `VK_NV_cooperative_vector` extension (minimum NVIDIA RTX 20XX) **|** [Vulkan SDK 1.3.296.0](https://vulkan.lunarg.com/sdk/home) **|** [NVIDIA public R570 driver or newer](https://www.nvidia.com/en-us/drivers/)

CMake detects the Vulkan SDK through the `VULKAN_SDK` environment variable set by its installer (or system-wide Vulkan headers on Linux). When it is not found, the Vulkan backend and the SPIR-V shaders are not built and the samples are DirectX only. Setting `-DDONUT_WITH_VULKAN=ON` without the SDK is a configuration error; `-DDONUT_WITH_VULKAN=OFF` skips Vulkan even when the SDK is installed.

\*Downloaded automatically by CMake during configuration; no separate installation is required.

## Known Issues

05/30/2025: When updating from v1.0.0 to v1.1.0, delete the CMake cache to avoid build errors.

## Project structure

| Directory                         | Details                                |
| --------------------------------- | -------------------------------------- |
| [/assets](assets)                 | _Asset files for samples_              |
| [/docs](docs)                     | _Documentation for showcased tech_     |
| [/samples](samples)               | _Samples showcasing usage of MLPs_     |
| [/external/donut](external/donut) | _Framework used for the examples_      |
| [/external](external)             | _Helper dependencies for the examples_ |
| [/src](src)                       | _Helper and utility functions_         |

## Getting started

- [Quick start guide](docs/QuickStart.md) for building and running the neural shading samples.
- [Library usage guide](docs/LibraryGuide.md) for using helper functions.

### External Resources

This project uses [Slang](https://shader-slang.com) and the Vulkan CoopVector extensions. The following links provide more detail on these, and other technologies which may help the reader to better understand the relevant technologies, or just to provide further reading.

* [Slang User Guide](https://shader-slang.com/slang/user-guide/)
  
  * [Automatic Differentiation](https://shader-slang.com/slang/user-guide/autodiff.html)

* [SlangPy](https://slangpy.readthedocs.io/en/latest/) 

* [Vulkan `VK_NV_cooperative_vector` extension](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_NV_cooperative_vector.html)

* [Donut](https://github.com/NVIDIAGameWorks/donut)

## Contact

RTXNS is actively being developed. Please report any issues directly through the GitHub issue tracker, and for any information or suggestions contact us at rtxns-sdk-support@nvidia.com

## Citation

Use the following BibTex entry to cite the usage of RTXNS in published research:

```bibtex
@online{RTXNS,
   title   = {{{NVIDIA}}\textregistered{} {RTXNS}},
   author  = {{NVIDIA}},
   year    = 2025,
   url     = {https://github.com/NVIDIA-RTX/RTXNS},
   urldate = {2025-02-03},
}
```

## License

See [LICENSE.md](LICENSE.MD)

[VisualStudio]: https://visualstudio.microsoft.com/

[Ninja]: https://ninja-build.org/

[CMake]: https://cmake.org/download/
