# RTX Neural Shading Change Log

## 1.6.0
- Kept SDK-managed dependency downloads and host tool builds within the SDK's directories, preserved explicit compiler paths, and prepared the DXC runtime for SDK-managed Slang in embedded builds.
- Replaced the legacy `ENABLE_DX12_COOP_VECTOR_PREVIEW` opt-in with `DONUT_WITH_DX12`, enabled by default on Windows and respecting the parent project's backend selection.
- Fixed embedded DX12 builds to reuse existing Agility SDK and NVAPI targets without replacing the parent's configuration.
- Updated Slang to `2026.20` and migrated to native `CoopVec` autodiff support. Slang `2026.20` or newer is required; see Known Issues in the README.
- The activation helpers in `CooperativeVectorFunctions.slang` are `[Differentiable]` and rely on the derivatives Slang synthesizes for `CoopVec`; no derivatives are registered by hand anymore.
- Improved half-precision gradient stability for the sigmoid, swish and tanh activations.
- Updated the SlangPy training sample to SlangPy `0.43.1`, using tensors and automatic backward-pass generation. `IModule` now extends `IDifferentiablePtrType`, mirroring `TrainingMLP`, so a module's backward pass is synthesized.
- Switched the SlangPy inferencing sample to Matplotlib display to fix window creation failures on Linux.
- The CMake project can be embedded with `add_subdirectory()` by a parent that already provides donut, NVRHI and ShaderMake; the SDK only forces toolchain and output settings when it is the top-level project (`RTXNS_IS_TOP_LEVEL`).
- The Slang `2026.20` Windows package no longer bundles DXC; the configure step places the pinned DXC (`v1.10.2605.24`) next to `slangc`.
- Removed `CooperativeVectorAutoDiff.slang` and `CooperativeVectorDerivatives.slang`. The `HCoopVec` alias moved to `CooperativeVectorFunctions.slang`; the `SimpleTraining` sample retains the derivative helpers needed by its manual backward pass.

## 1.5.0
- Removed the legacy DX12 Cooperative Vector preview `717` / Shader Model 6.9 toolchain path.
- Updated the DX12 driver requirement for the `721` toolchain to the public NVIDIA R615 driver or newer; the preview driver is no longer required.
- Updated the Agility SDK to `1.721.3-preview`.
- Added a project-side workaround for invalid enhanced-barrier validation during DX12 matrix conversion.
- Added Arm64 (Windows on Arm) build support. Architecture-specific dependencies are selected for the target or build host as appropriate, and ShaderMake is built for the host when cross-compiling.
- Updated NVAPI to the `R615` developer SDK, including its Arm64 and Arm64EC libraries. NVAPI is downloaded as a source archive of the pinned commit instead of a full Git clone.
- Build outputs are now written per configuration to `bin/<platform>/<configuration>/`, so Debug and Release builds no longer overwrite each other.
- Added Ninja-based Windows x64 and Arm64 Debug and Release presets, plus Linux presets. The Windows toolchain files support native and cross builds and validate the selected compiler environment. Other CMake generators remain supported for custom builds.
- `DONUT_WITH_VULKAN` now defaults to whether the Vulkan SDK is installed, and enabling it without the SDK is a configure-time error. Without the SDK the Vulkan backend and SPIR-V shaders are skipped instead of compiling SPIR-V with whichever `dxc` is found; cross-compiled builds previously picked up the Windows SDK's DXC, which has no SPIR-V code generation (`SPIR-V CodeGen not available`).
- Vulkan-Headers and DirectX-Headers are now downloaded once as tag archives into shared, versioned `external/` folders instead of being cloned separately for every build directory.
- Updated the minimum CMake version to 3.24 and consolidated shared archive downloads. DXC, Slang, and headers are reused across build directories, host tool overrides are supported, and generated Slang launcher scripts are platform-specific.

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
