# RTX Neural Shading

RTX Neural Shading (RTXNS) also known as RTX Neural Shaders, is intended as a starting point for developers interested in bringing Machine Learning (ML) to their graphics applications. It provides a number of examples to help the reader understand how to train their own neural networks and then use those models to perform inference alongside their normal graphics rendering. 

RTXNS uses the [Slang](https://shader-slang.com) shading language and it utilizes either the DirectX Preview Agility SDK or the Vulkan Cooperative Vectors extension to provide access to the GPUs ML acceleration.

A number of examples are included which build upon each other from a simple inference example to more complex examples showing how to train a neural network to represent a shader or a texture. Helper functions to facilitate building your own neural networks are also included. 

Alongside the core samples is a SlangPy sample to demonstrate how to use python and SlangPy for fast iteration and development of neural networks which can then be integrated into RTXNS for inference. 

When exploring RTXNS, it is assumed that the reader is already familiar with ML and neural networks.

## Requirements

### General
[CMake v3.24.3][CMake] **|** [VS 2022][VS22] **|** [Slang v2025.10](https://shader-slang.com/tools/)

### DirectX
[DirectX Preview Agility SDK 1.717.0-preview](https://www.nuget.org/packages/Microsoft.Direct3D.D3D12/1.717.0-preview) **|** [Microsoft DXC 1.8.2505.28](https://www.nuget.org/packages/Microsoft.Direct3D.DXC/1.8.2505.28) **|** [Geforce Shader Model 6-9-Preview Driver](https://developer.nvidia.com/downloads/shadermodel6-9-preview-driver)  **|** [Quadro Shader Model 6-9-Preview Driver](https://developer.nvidia.com/downloads/assets/secure/shadermodel6-9-preview-driver-quadro) 

### Vulkan
GPU must support the Vulkan `VK_NV_cooperative_vector` extension (minimum NVIDIA RTX 20XX) **|** [Vulkan SDK 1.3.296.0](https://vulkan.lunarg.com/sdk/home) **|** Public Driver ≥ 572.16

## Known Issues
05/30/2025: When updating from v1.0.0 to v1.1.0 is it recommended to delete the cmake cache to avoid build errors.

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
- [Library usage guide](docs/LibraryGuide.md) for using helper functions

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

[VS22]: https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&passive=false&cid=2030

[CMake]: https://github.com/Kitware/CMake/releases/download/v3.24.3/cmake-3.24.3-windows-x86_64.msi
