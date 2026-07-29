# RTX Neural Shading: Quick Start Guide
RTX Neural Shading can be build and run on both Windows and Linux

## Build steps

1. Clone the project recursively:
   
   ```
   git clone --recursive https://github.com/NVIDIA-RTX/RTXNS
   ```

2. Create a build directory:
   
   ```
   cd RTXNS
   mkdir build

   ```
3. Configure the build using your preferred CMake generator.

   ```
   cmake -S . -B build -G <generator>
   ```

   To enable the DX12 Cooperative Vector preview set the option `ENABLE_DX12_COOP_VECTOR_PREVIEW` on (Windows only).
   ```
   cmake -DENABLE_DX12_COOP_VECTOR_PREVIEW=ON
   ```

4. Open build/RtxNeuralShading.sln in Visual Studio and build all projects, or build using the CMake CLI:
   
   ```
   cmake --build build --config Release

   ```

5. All of the sample binaries can be found in `/bin` such as
   
   ```
   bin/<platform>/SimpleInferencing
   ```

6. The samples can be launched as either DX12 or Vulkan where supported with the respective commandline: `-dx12` or `-vk` 

## About

All of the samples are built using Slang and can be compiled to either DX12 or Vulkan using DirectX Preview Agility SDK or Vulkan Cooperative Vector extension respectively. 

- [DirectX Preview Agility SDK](https://devblogs.microsoft.com/directx/directx12agility/).
- [Vulkan Cooperative Vector extension](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_NV_cooperative_vector.html).

## Driver Requirements
- The default DirectX configuration uses Agility SDK `1.721.2-preview` with Shader Model 6.10 and requires a compatible [GeForce preview driver](https://developer.nvidia.com/downloads/assets/secure/geforce-drivers/620.12_gameready_win11_win10-dch_64bit_international.exe).
- The legacy `717` / Shader Model 6.9 configuration can be selected with `-DRTXNS_DX12_COOP_VECTOR_PREVIEW=717` and requires a compatible preview driver:
	- [GeForce](https://developer.nvidia.com/downloads/shadermodel6-9-preview-driver)  
	- [Quadro](https://developer.nvidia.com/downloads/assets/secure/shadermodel6-9-preview-driver-quadro)
- Vulkan Cooperative Vector extension requires a release [driver](https://www.nvidia.com/en-gb/geforce/drivers) from R570 onwards

### Samples

| Sample Name                                | Output                                                                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Simple Inferencing](SimpleInferencing.md) | [<img src="simple_inferencing.png" width="800">](simple_inferencing.png) | This sample demonstrates how to implement an inference shader using some of the low-level building blocks from RTXNS. The sample loads a trained network from a file and uses the network to approximate a Disney BRDF shader. The sample is interactive; the light source can be rotated and various material parameters can be modified at runtime.                                                                                                      |
| [Simple Training](SimpleTraining.md)       | [<img src="simple_training.png" width="800">](simple_training.png)       | This sample builds on the Simple Inferencing sample to provide an introduction to training a neural network for use in a shader. The network replicates a transformed texture.                                                                                                                                                                                                                                                                             |
| [Shader Training](ShaderTraining.md)       | [<img src="shader_training.png" width="800">](shader_training.png)       | This sample extends the techniques shown in the Simple Training example and introduces Slangs AutoDiff functionality, via a full MLP (Multi Layered Perceptron) abstraction. The MLP is implemented using the `CoopVector` training code previously introduced and provides a simple interface for training networks with Slang. The sample creates a network and trains a model on the Disney BRDF shader that was used in the Simple Inferencing sample. |
| [SlangPy Training](SlangpyTraining.md)     | [<img src="slangpy_training.jpg" width="800">](slangpy_training.jpg)     | This sample shows how to create and train network architectures in python using SlangPy. This lets you experiment with different networks, encodings and more using the building blocks from RTXNS, but without needing to change or rebuild C++ code. As a demonstration this sample instantiates multiple different network architectures and trains them side-by-side on the same data. It also shows one possible approach of exporting the network parameters and architecture to disk so it can be loaded in C++. |
| [SlangPy Inferencing](SlangpyInferencing.md) | [<img src="slangpy_inferencing_window.png" width="800">](slangpy_inferencing_window.png) | This sample demonstrates how to run neural network inference in Python using the SlangPy library and then transition the same implementation to C++. The workflow illustrates a typical development pattern where initial prototyping and experimentation is done in Python using SlangPy for its flexibility and ease of use, and the same Slang code is later deployed in a C++ application for production use. The sample includes both Python and C++ implementations that perform the same neural network inference task, providing a clear path for transitioning between the two environments. |

### Tutorial

* [Tutorial](Tutorial.md) 
  A tutorial to help guide you to create your own neural shader based on the [Shader Training](ShaderTraining.md) example.

### Library

* [Library](LibraryGuide.md) 
  A guide to using the library / helper functions to create and manage your neural networks.
