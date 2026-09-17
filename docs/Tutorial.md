# RTX Neural Shading: How to Write Your First Neural Shader

## Purpose

Using  [Shader Training](ShaderTraining.md) as the basis of this tutorial, we will briefly discuss an approach to writing your first neural shader.

The main areas we will focus on are :

1. Extracting the key features from the shader to be trained

2. Modifying the network configuration

3. Modifying the activation and loss functions

It is outside the scope of this document to discuss how AI training and optimization works and instead we will focus on modifying the existing sample to configure and train the network with different content.

## Extracting the Key Features for Training Input

When implementing the Disney BRDF for the [Shader Training](ShaderTraining.md) example, the first task was feature extraction: deciding which shader features the network should infer and which should remain explicit calculations so that the network is neither overspecialized nor unnecessarily complex. The Disney BRDF network derives inputs from the `view`, `light`, and `normal` vectors, together with `material roughness`. Variables such as `light intensity`, `material metallicness`, and material-color components remain in the shader. Finding the right balance may require experimentation.

Once the key features are identified, simplify them where possible and scale them to `[0, 1]` or `[-1, 1]`. In the Disney BRDF, the input vectors are normalized and used only through dot products, so three `float3` vectors are reduced to four scalar dot products.

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
  
Precision affects both quality and performance. The Shader Training sample is configured for `float16`; changing precision requires updating the matching C++ network architecture, buffer formats, and Slang types.

Changing any of these parameters should not require any further code changes as the defines are shared amongst the C++ and shader code; they will just require a re-compile.  The exception may be when changing the size of input/output `CoopVecs`  and any code that dereferences their elements directly, such as :

```
float4 predictedDisney = { outputParams[0], outputParams[1], outputParams[2], outputParams[3] };
```

As always, experimentation will be required to find the right set of configuration parameters for the optimal training of your network.

## Modifying the Activation and Loss Functions

The Shader Training example uses `TrainingMLP`, which abstracts much of the training shader code for the user:

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

The activation functions are passed into the model's forward and backward passes (`ReLUAct` and `ExponentialAct`) for use with `TrainingMLP` and `InferenceMLP`. They are defined in [Activation.slang](../src/NeuralShading_Shaders/Activation.slang) and can be extended as necessary.

The choice of loss function depends on your dataset. The Simple Training example uses an L2 loss function, whereas the Shader Training example uses a more complex relative L2 loss. Custom loss functions can be implemented in Slang to help tune a shader.

## Hyperparameters

These are some of the hyperparameters available for tuning a dataset.

| Parameter                   | Value            |
| --------------------------- | ---------------- |
| HIDDEN_NEURONS              | 32               |
| NUM_HIDDEN_LAYERS           | 3                |
| LEARNING_RATE               | 1e-3f            |
| BATCH_SIZE                  | (1 << 16)        |
| BATCH_COUNT                 | 100              |
| Hidden Activation Functions | ReLUAct()        |
| Final Activation Functions  | ExponentialAct() |
| Loss Function               | L2Relative()     |

## Summary

The Shader Training sample is a good place to start to train your own neural shader. It will require some thought as to how to decompose your shader into network inputs and shader inputs and then the network can be re-configured through experimentation to find the suitable model that can handle your dataset.
