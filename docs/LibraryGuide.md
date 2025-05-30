# RTX Neural Shading: Library Usage Start Guide

The library is split into 2 main areas - the application side and the shader side. 

The application part of the library contains a suite of helper functions to create neural networks, serialize them to/from disk, change their precision and layout as well allocate and destroy the required backing storage. These utilize the `nvrhi` SDK to provide a graphics agnostic interface, but can easily be changed to suit a different engine.

The shader part of the library contains the necessary Slang helper functions needed for training and running inference from a small neural network.

## Application Code

The main utility classes for creating neural networks can be found in `NeuralNetwork.h`, which contains `rtxns::HostNetwork` and  `rtxns::NetworkUtilities`. `rtxns::HostNetwork` wraps host allocation for the weights and biases with functions to store/load the network to file, where `rtxns::NetworkUtilities` contains functions to convert between a host matrix layout and device optimal matrix layout.

The `rtxns::HostNetwork`  object must be created and initialized before use. It can be initialized from input parameters describing the network architecture, from a file or from another `rtxns::Network` . In each of these cases the network will be in a host layout (`rtxns::MatrixLayout::RowMajor` or `rtxns::MatrixLayout::ColumnMajor`). 

```
// Initialise an empty network from parameters
nvrhi::IDevice* device = ...
rtxns::HostNetwork hostNetwork = rtxns::HostNetwork(device);

rtxns::NetworkArchitecture netArch = {};
netArch.inputNeurons = 2;
netArch.hiddenNeurons = 32;
netArch.outputNeurons = 3;
netArch.numHiddenLayers = 3;
netArch.biasPrecision = rtxns::Precision::f16;
netArch.weightPrecision = rtxns::Precision::f16;

if (!hostNetwork.Initialise(netArch))
    log::error("Failed to create a network from an arch!");
```

```
// Initialise a network from a file
nvrhi::IDevice* device = ...
rtxns::HostNetwork hostNetwork = rtxns::Network(device);
if (!hostNetwork.InitialiseFromFile("myNN.bin"))
    log::error("Failed to create a network from myNN.bin!");
```

Creating the network will allocate the required host side memory to store the weights and biases per layer of the network. It will not allocate any GPU memory, but instead, it provides the required size and offsets so the user can make their own GPU allocations and copy over the host data as required.

The host weights and biases are correctly sized for a direct copy to the GPU. They are accessed via a network parameters accessor and the offsets are queried from the layer accessor :

```
const std::vector<uint8_t>& params = neuralNetwork.GetNetworkParams();

// Copy to GPU buffer
copy(hostBuffer, params.data(), params.size());
```

The network has the notion of the underlying matrix layout, these are categorized into either a host layout or a device optimal layout. Note that the host layout can be used on the GPU but may not be as performant as the device optimal layouts.

```
enum class MatrixLayout
{
    RowMajor,
    ColumnMajor,
    InferencingOptimal,
    TrainingOptimal,
};
```

The host layouts are `rtxns::MatrixLayout::RowMajor` and `rtxns::MatrixLayout::ColumnMajor` as they are both hardware and API agnostic, this makes them suitable for storing a network to a file. The device optimal layouts are `rtxns::MatrixLayout::InferencingOptimal` and `rtxns::MatrixLayout::TrainingOptimal`, which are opaque HW specific formats that are not guaranteed to be transferable between GPUs and APIs and will often have specific data alignment and padding requirements.

The typical lifecycle of a network would start in a host layout, where it is set to either initial values or loaded from file. This network would then be upload to the GPU and converted into a device optimal layout using `rtxns::NetworkUtilities::ConvertWeights`. `rtxns::MatrixLayout::TrainingOptimal` would be used whilst training the network and again using `rtxns::NetworkUtilities::ConvertWeights` changed to `rtxns::MatrixLayout::InferenceOptimal` when that training is complete. To store the trained network we must convert back to a host layout before writing to a file so it can be shared between GPUs.

To change the layout of the network we first use `rtxns::NetworkUtilities::GetNewMatrixLayout()` to create a network layout of the same architecture as the host that is device optimal. The weight and bias offsets are stored for the device layout and will be passed to the GPU via constant buffer.

```
// Get a device optimized layout
rtxns::NetworkLayout deviceNetworkLayout = m_networkUtils->GetNewMatrixLayout(neuralNetwork.GetNetworkLayout(), rtxns::MatrixLayout::TrainingOptimal);

// Store the device layout offsets 
weightOffsets = dm::uint4(
    deviceNetworkLayout.networkLayers[0].weightOffset,
    deviceNetworkLayout.networkLayers[1].weightOffset,
    deviceNetworkLayout.networkLayers[2].weightOffset,
    deviceNetworkLayout.networkLayers[3].weightOffset);
biasOffsets = dm::uint4(
    deviceNetworkLayout.networkLayers[0].biasOffset,
    deviceNetworkLayout.networkLayers[1].biasOffset,
    deviceNetworkLayout.networkLayers[2].biasOffset,
    deviceNetworkLayout.networkLayers[3].biasOffset);

```

After creating device side buffer of `deviceNetworkLayout.networkSize` we then use `rtxns::NetworkUtilities::ConvertWeights()` to convert the host layout to the device optimal layout.
```
ConvertWeights(hostNetwork.GetNetworkLayout(),
    deviceNetworkLayout,
    hostBuffer,
    hostOffset,
    deviceBuffer,
    deviceOffset,
    device,
    commandList);
```
The device side buffer is now ready for training. The above steps of `rtxns::NetworkUtilities::GetNewMatrixLayout()` and `rtxns::NetworkUtilities::ConvertWeights()` are required again when changing from `rtxns::MatrixLayout::TrainingOptimal` to `rtxns::MatrixLayout::InferencingOptimal`
The trained network can also be convert back to a host side layout with `rtxns::NetworkUtilities::ConvertWeights` so it can be written to file. This functionality is wrapped up in `rtxns::HostNetwork::UpdateFromBufferToFile()`.

### Cooperative Vectors

If the user wants to explore writing their own neural network class, then they should investigate the `ICoopVectorUtils` class in `CoopVector.h` and its usage within the `NeuralNetwork.h` described above. It provides an API agnostic interface to the Vulkan and DX12 Cooperative Vector extension that allows the user to query matrix sizes and convert device data between layouts and supported precisions on the GPU.

## Shader Code

The shader code library is split into several sections

### Linear Operations Module

The linear operations module contains the main functions for running inferencing and training; `LinearOp` and `LinearOp_Backward`. The module also contain a backward derivative implementation `LinearOp` that can be used with Slang autodiff feature.

The `LinearOp` function is used to carry out a forward linear step in a neural network from an input layer of size `K` to the next layer of size `M`, where the weight and bias are stored in a single buffer. `CoopVecMatrixLayout` states the layout the weight matrix being used which should match the `MatrixLayout` set on the C++ side. `CoopVecComponentType` determines how the matrix should be interpreted, which in most cases should match the type `T`.

```
CoopVec<T, M> LinearOp<T : __BuiltinFloatingPointType, let M : int, let K : int>( 
    CoopVec<T, K> ip, 
    ByteAddressBuffer matrixBiasBuffer, 
    uint matrixOffset, 
    int biasOffset, 
    constexpr CoopVecMatrixLayout matrixLayout, 
    constexpr CoopVecComponentType componentType)
```

The `LinearOp_Backward` function is used to carry out a backwards linear step in a neural network applying a gradient of size `M` to the previous layer of size `K`. The  weight, bias and their derivatives are each stored in their respective buffer. As with `LinearOp`, `CoopVecMatrixLayout` states the layout of the weight matrix being used and  `CoopVecComponentType` determines how the matrix should be interpreted.

```
 CoopVec<T, K> LinearOp_Backward<T : __BuiltinFloatingPointType, let M : int, let K : int>(
    CoopVec<T, K> ip, 
    CoopVec<T, M> grad, 
    ByteAddressBuffer matrixBiasBuffer, 
    RWByteAddressBuffer matrixBiasBufferDerivative, 
    uint matrixOffset, 
    int biasOffset, 
    constexpr CoopVecMatrixLayout matrixLayout, 
    constexpr CoopVecComponentType componentType)
```

#### Differentiable  LinearOps

The second half of this module extends the functionality of cooperative vectors to provide support for Slang's auto differentiation feature as it is not natively supported. 

The `MatrixBiasBuffer` and `MatrixBiasBufferDifferential` structures inherit Slang's `IDifferentiablePtrType` interface so the matrix buffer and its derivative will support auto differentiation.

```
struct MatrixBiasBufferDifferential : IDifferentiablePtrType
    {
        typealias Differential = MatrixBiasBufferDifferential;

        __init(RWByteAddressBuffer buf) 
        { 
            buffer = buf;
        }

        RWByteAddressBuffer buffer;
};

struct MatrixBiasBuffer : IDifferentiablePtrType
    {
        typealias Differential = MatrixBiasBufferDifferential;

        __init(ByteAddressBuffer buf) 
        { 
            buffer = buf;
        }

        ByteAddressBuffer buffer;
};
```

Next we have a differentiable version of `LinearOp` where the `matrixBiasBuffer` is replaced with the `MatrixBiasBuffer` struct and the offsets are passed in. 

```
CoopVec<T, M> LinearOp<T : __BuiltinFloatingPointType, let M : int, let K : int>( 
    CoopVec<T, K> ip, 
    MatrixBiasBuffer matrixBiasBuffer, 
    uint2 offsets,
    constexpr CoopVecMatrixLayout matrixLayout, 
    constexpr CoopVecComponentType componentType)
```

`LinearOp_BackwardAutoDiff` is the backwards derivative of LinearOp, where the input `CoopVec` and `MatrixBiasBuffer` are now passed in a `DifferentialPair`. See Slang documentation for details

```
[BackwardDerivativeOf(LinearOp)]
void LinearOp_BackwardAutoDiff<T : __BuiltinFloatingPointType, let M : int, let K : int>( 
    inout DifferentialPair<CoopVec<T, K>> ip, 
    DifferentialPtrPair<MatrixBiasBuffer> MatrixBiasBuffer, 
    uint2 offsets,
    constexpr CoopVecMatrixLayout matrixLayout, 
    constexpr CoopVecComponentType componentType, 
    CoopVec<T, M>.Differential grad)
```

### Activation Function Module

This module provides implementations of common activation functions

```
struct NoneAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>

struct LinearAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>

struct ExponentialAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>

struct ShiftedExponentialAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>

struct ReLUAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>

struct LeakyReLUAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>

struct SigmoidAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>

struct SwishAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>

struct TanhAct<T : __BuiltinFloatingPointType, let K : int> : IActivation<T, K>

```

### MLP Module

The MLP module contains two structures for representing and running a neural network; `InferenceMLP` and `TrainingMLP`. Both structures contain functions for running a full forward pass on the network, where they differ is the use case for each. `InferenceMLP` is for inferencing only and provides a forward pass function only. `TrainingMLP` is used for training a network and contains an additional buffer for derivatives and a backwards pass function. `TrainingMLP` uses Slang's auto differentiation  functionality to generate a backwards propagation function instead of providing an implementation for it. 

#### InferenceMLP

```
 struct InferenceMLP<
        T : __BuiltinFloatingPointType, 
        let HIDDEN_LAYERS : int, 
        let INPUTS : int, 
        let HIDDEN : int, 
        let OUTPUTS : int, 
        let matrixLayout : CoopVecMatrixLayout, 
        let componentType : CoopVecComponentType
    >
    {
        ...
        CoopVec<T, OUTPUTS> forward<Act : IActivation<T, HIDDEN>, FinalAct : IActivation<T, OUTPUTS>>(
            CoopVec<T, INPUTS> inputParams, 
            Act act, 
            FinalAct finalAct);
        ...
    }
```

#### TrainingMLP

```
 struct TrainingMLP<
        T : __BuiltinFloatingPointType, 
        let HIDDEN_LAYERS : int, 
        let INPUTS : int, 
        let HIDDEN : int, 
        let OUTPUTS : int, 
        let matrixLayout : CoopVecMatrixLayout, 
        let componentType : CoopVecComponentType
    >
    {
        ...
        CoopVec<T, OUTPUTS> forward<Act : IActivation<T, HIDDEN>, FinalAct : IActivation<T, OUTPUTS>>(CoopVec<T, INPUTS> inputParams, Act act, FinalAct finalAct);

        void backward<Act : IActivation<T, HIDDEN>, FAct : IActivation<T, OUTPUTS>>(CoopVec<T, INPUTS> ip, Act act, FAct fact, CoopVec<T, OUTPUTS> loss);
        ...
    }
```

### Optimizer Module

This module provides an interface and implementations of common optimizer functions

The interface consists of step functions required for each implementation

```
interface IOptimizer
{
    float step(float weightBias, uint parameterID, float gradient, const float currentStep);
};
```

The module contains an implementation of the Adam optimizer algorithm, which add two moment buffers and hyper parameters

```
struct Adam : IOptimizer
{
    RWBuffer<float> m_moments1;
    RWBuffer<float> m_moments2;
    float m_learningRate;
    float m_lossScale;
    float m_beta1;
    float m_beta2;
    float m_epsilon;
}
```

### Utility Module

This module provides functionality for input encoding and packing weight and bias buffer offsets

The encoder functions can be use to increase the input count of a neural network, providing additional information to assist the learning process. Use of these should be validated to confirm they improve quality and / or performance.

`CoopVecFromArray` simply constructs a `CoopVec`of matching size from a float array.

```
CoopVec<T, PARAMS_COUNT> CoopVecFromArray<T : __BuiltinFloatingPointType, let PARAMS_COUNT : int>(float parameters[PARAMS_COUNT])
```

`EncodeFrequency` expands the input parameters by 6 for each input, which are encoded with sine and cosine waves

```
CoopVec<T, PARAMS_COUNT * FREQUENCY_ENCODING_COUNT> EncodeFrequency<T : __BuiltinFloatingPointType, let PARAMS_COUNT : int>(float parameters[PARAMS_COUNT])
```

`EncodeTriangle` similar to frequency encoding this expands the input parameters by 6 for each input, encoding them to represent a triangle wave

```
CoopVec<T, PARAMS_COUNT * TRIANGLE_ENCODING_COUNT> EncodeTriangle<T : __BuiltinFloatingPointType, let PARAMS_COUNT : int>(float parameters[PARAMS_COUNT])
```

The `UnpackArray` function is used to unpack the weight and bias offsets from a constant buffer aligned uint4 array

```
uint[NUM_UNPACKED] UnpackArray<let NUM_PACKED4 : int, let NUM_UNPACKED : int>(uint4 ps[NUM_PACKED4])
```
