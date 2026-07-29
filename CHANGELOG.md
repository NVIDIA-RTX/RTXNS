# RTX Neural Shading Change Log

## 1.4.0
- Added support for the DX12 Cooperative Vector / Linear Algebra preview `721` toolchain, with temporary compatibility support for preview `717`.
- Updated shader toolchain defaults to Slang `2026.10`, DXC `v1.10.2605.24`, Agility SDK `1.721.2-preview`, and Shader Model `6_10`.
- Updated DirectX setup documentation and release links for Agility SDK `1.721.2-preview` and DXC `v1.10.2605.24`.
- Updated cooperative vector shader code for stricter Slang autodiff rules.
- Improved cooperative vector feature detection using newer NVRHI query APIs.
- Centralized sample shader compile options and enabled embedded shader PDBs.
- Updated the SlangPy inferencing sample to SlangPy `0.43.1`, including current tensor and cooperative-vector APIs, correct Vulkan buffer usage, and updated swapchain synchronization.
- Updated Donut/NVRHI integration.

## 1.3.1
- Updated documentation to include Linux instructions.
- Migrate `createGraphicsPipeline` from deprecated function to the current NVRHI API.
- Update to `ResultsWidget` to allow for user configuration of data size and graph ranges. 
- Bug Fixes
	- Corrected `L2Relative` loss function.

## 1.3.0
- Added loss visualization graphs to the `ShaderTraining` and `SimpleTraining` examples.
- Added ImPlot support, enabling real-time plotting for diagnostics and training visualization.
- Introduced helper utilities for tracking and accumulating loss each epoch.
- Bug Fixes
	- Fixed incorrect buffer sizes in `ShaderTraining`.
	- Resolved potential issues when loading and storing networks.	
- Updates
	- Updated Slang to version `2025.23.1`.	 
## 1.2.2
- Fixes bug in matrix alignment
- Updated error messages for DX12

## 1.2.1 
- Added learning rate scheduler with cosine decay for smoother convergence over long training runs.
- Simple Training
	- Improved training stability with learning rate scheduler
	- Added UI display to show the current learning rate during training.

## 1.2.0
- Added Slangpy Inferencing sample.
	- This sample demonstrates how to deploy a neural network prototyped with Python and SlangPy using C++ and Slang.
- Updated samples to use NVRHI for Cooperative Vector format queries and matrix conversions.

## 1.1.0
- Added DX12 cooperative vector support using Preview Agility SDK.
- Moved matrix conversion to GPU.

## 1.0.0
- Initial release.
