# How to Write Your First Neural Shader

## Purpose

Using  [Shader Training](ShaderTraining.md) as the basis of this tutorial, we will briefly discuss an approach to writing your first neural shader.

The main areas we will focus on are :

1. Extracting the key features from the shader to be trained

2. Modifying the network configuration

3. Modifying the activation and loss functions

It is outside the scope of this document to discuss how AI training and optimisation works and instead we will focus on modifying the existing sample to configure and train the network with different content.

## Extracting the Key Features for Training Input

When implementing the Disney BRDF for use in the [Shader Training](ShaderTraining.md) example, the first task was feature extraction. Which features from the shader should be infered from the network and which should be calculated to ensure the network is not over specialised or overly complex. The network for the Disney BRDF takes inputs such as the `view`, `light` and `normal` vectors as well as  `material roughness`. Other variables, such as `light intensity`, `material metallicness` and various `material colour` components have been left as part of the shader. This is a balancing act which may require some experimentation.

Once the key features are identified as potential training inputs, look to optimise them where possible by reducing their form and scaling them to be in the range `0-1` or `-1 - 1` which is preferred by networks. In the Disney BRDF, this was done by recognising that the input vectors where always normalised and used in their dot product form, so the inputs were reduced from 3 `float3` vectors, to 4 `float` dot products.

Next, the network inputs may benefit from encoding which research has shown to improve the performance of the network. The library provides 2 encoders, `EncodeFrequency` and `EncodeTriangle` which encode the inputs into some form of wave function. The shader training example uses the frequency encoder which increases the number of inputs by a factor of 6 but provides a better network as a result. You should experiment with encoders to find the one suitable for your dataset.

At this point, you should know the number of (encoded) input parameters and output parameters, so it is time to configure the network.

## Modifying the Network Configuration

The network configuration is stored in [NetworkConfig.h](../samples/ShaderTraining/NetworkConfig.h), and may require modification. Some elements are fixed for your dataset, like the input and output neuron counts and others are available for configuration. In the provided samples, the configuration is hard-coded for ease of understanding, but in a production system they are fully expected to be a configurable part of the training pipeline.

These are fixed configuration parameters that are directly tied to the shader you are trying to train from :

- `INPUT_NEURONS` should equal the number of encoded input parameters from above that are directly passed into the network.

- `OUTPUT_NEURONS` should equal the output parameters that the network generates. This may be an RGB triple, or just a number of unconnected outputs like for the DisneyBRDF.
  
The following parameters are available for experimentation and should be modified to find suitable settings for the network you are trying to train :

- `NUM_HIDDEN_LAYERS` - The number of hidden layers that make up the network.

- `HIDDEN_NEURONS` - The number of neurons in the hidden layers of the network. Changing this can make significant differences to the accuracy and cost of your network.

- `LEARNING_RATE` - This should be tuned to improve convergence of your model.
  
In future versions of the library, precision of the neurons may be alterable which could change the quality and performance of the network. The current version is fixed to `float16`.

Changing any of these parameters should not require any further code changes as the defines are shared amongst the C++ and shader code; they will just require a re-compile.  The exception may be when changing the size of input/output `CoopVecs`  and any code that dereferences their elements directly, such as :

```
float4 predictedDisney = { outputParams[0], outputParams[1], outputParams[2], outputParams[3] };
```

As always, experimentation will be required to find the right set of configuration parameters for the optimal training of your network.

## Modifying the Activation and Loss Functions

The Simple Shading example uses the `TrainingMLP` which abstracts much of the training shader code for the user :

```
var model = rtxns::mlp::TrainingMLP<half, 
    NUM_HIDDEN_LAYERS, INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS, 
    CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16>(
    gMLPParams, 
    gMLPParamsGradients, 
    rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets),
    rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets));

var hiddenActivation = rtxns::mlp::ReLUAct<half, HIDDEN_NEURONS>();
var finalActivation = rtxns::mlp::ExponentialAct<half, OUTPUT_NEURONS>();

var outputParams = model.forward(inputParams, hiddenActivation, finalActivation);
```

The activation functions are passed into the models forward and backward pass (`ReLUAct` and `ExponentialAct`) for use with the `TrainingMLP` and `InferenceMLP`. These can be found in [CooperativeVectorFunctions.slang](../src/NeuralShading_Shaders/CooperativeVectorFunctions.slang) and extended as necessary. The current version of RTXNS provides a limited set of activation functions, but these can be examined and modified to support more activation functions as required.

The choice of loss function to use will be dependant on your dataset. The Simple Training example uses a simple L2 loss function whereas the Shader Training example uses a more complex L2 relative loss function. Any loss function can be trivially coded in slang to help tune your shader.

## Hyper Parameters

These are some of the hyper parameters that are available for tuning for your dataset.

| Parameter                   | Value            |
| --------------------------- | ---------------- |
| HIDDEN_NEURONS              | 32               |
| NUM_HIDDEN_LAYERS           | 3                |
| LEARNING_RATE               | 1e-2f            |
| BATCH_SIZE                  | (1 << 16)        |
| BATCH_COUNT                 | 100              |
| Hidden Activation Functions | ReLUAct()        |
| Final Activation Functions  | ExponentialAct() |
| Loss Function               | L2Relative()     |

## Summary

The Shader Training sample is a good place to start to train your own neural shader. It will require some thought as to how to decompose your shader into network inputs and shader inputs and then the network can be re-configured through experimentation to find the suitable model that can handle your dataset.
