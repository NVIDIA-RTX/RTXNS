#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import slangpy as spy

import numpy as np
import json
import math
import time
import sys

from Helpers import SDKSample
from NeuralModules import CoopVecModule, TrainableMLP, FrequencyEncoding, ModuleChain
from NeuralModules import Activation, NoneAct, LinearAct, ExponentialAct, ShiftedExponentialAct, ReLUAct, LeakyReLUAct, SigmoidAct, SwishAct, TanhAct

# Set to true for an interactive training. This can be helpful
# but slows down training quite a bit
INTERACTIVE = True
if INTERACTIVE:
    import matplotlib.pyplot as plt

def training_main():
    ##
    ## Setup window, device and file paths
    ##
    sample = SDKSample(sys.argv[1:])
    device = sample.device

    ##
    ## Set up training constants.
    ## When we train interactively, choose smaller batches
    ## for faster feedback.
    ##
    batch_shape = (256, 256)
    learning_rate = 0.005
    grad_scale = 128.0
    loss_scale = grad_scale / math.prod(batch_shape)

    sample_target = 1000000000
    num_batches_per_epoch = 1000 if INTERACTIVE else 5000
    num_epochs = sample_target // (num_batches_per_epoch * math.prod(batch_shape))

    ##
    ## Set up models
    ##

    # A basic MLP with ReLU activations and a linear output that maps a 2D UV input
    # to an RGB color. This is a good baseline, but it won't achieve state-of-the-art
    basic_mlp = TrainableMLP(device, spy.DataType.float16,
                             num_hidden_layers=3,
                             input_width=2,
                             hidden_width=32,
                             output_width=3,
                             hidden_act=ReLUAct(),
                             output_act=NoneAct())

    # Replacing ReLU with LeakyReLU makes training more stable for small networks,
    # and a Sigmoid activation at the output helps bring the network into the right range
    better_activations = TrainableMLP(device, spy.DataType.float16,
                                      num_hidden_layers=3,
                                      input_width=2,
                                      hidden_width=32,
                                      output_width=3,
                                      hidden_act=LeakyReLUAct(),
                                      output_act=SigmoidAct())

    # For 2D or 3D inputs, we can do even better with an input encoding
    # We need to adjust the input width of the MLP to take the additional
    # outputs from the encoding
    encoding = FrequencyEncoding(spy.DataType.float16, 2, 3)
    mlp_with_encoding = ModuleChain(
        encoding,
        TrainableMLP(device, spy.DataType.float16,
                     num_hidden_layers=3,
                     input_width=encoding.fan_out,
                     hidden_width=32,
                     output_width=3,
                     hidden_act=LeakyReLUAct(),
                     output_act=SigmoidAct())
    )

    # We're not limited to predefined modules - for example, try using the custom
    # activation from the slang file:
    activation = SigmoidAct()
    #activation = Activation("SiLUActivation")

    # Now take the working model and scale up the number of weights by adding another layer
    larger_mlp = ModuleChain(
        encoding,
        TrainableMLP(device, spy.DataType.float16,
                     num_hidden_layers=4,
                     input_width=encoding.fan_out,
                     hidden_width=32,
                     output_width=3,
                     hidden_act=LeakyReLUAct(),
                     output_act=activation)
    )

    # Make a list of models to be optimized so we can compare them
    models = [
        ("Basic MLP", basic_mlp),
        ("+Better activations", better_activations),
        ("+Frequency encoding", mlp_with_encoding),
        ("+More Weights", larger_mlp),
    ]

    # You can also play with different losses. For images, L2 is not a bad default
    loss_name = "rtxns::mlp::L2<float, 3>"

    ##
    ## Load training data and slang code
    ##
    target_tex = sample.load_texture("nvidia-logo.png")

    module = spy.Module.load_from_file(device, "SlangpyTraining.slang")

    # Instantiate the slang RNG from the loaded module,
    # seeded with a random buffer of uints
    pcg = np.random.PCG64(seed=12345)
    seeds = pcg.random_raw(batch_shape).astype(np.uint32)
    rng = module.RNG(seeds)

    # Fill a buffer with UVs for later evaluating the model during training
    vis_resolution = 256
    span = np.linspace(0, 1, vis_resolution, dtype=np.float32)
    vis_uvs_np = np.stack(np.broadcast_arrays(span[None, :], span[:, None]), axis=2)
    vis_uvs = spy.NDBuffer(device, module.float2.struct, shape=(vis_resolution, vis_resolution))
    vis_uvs.copy_from_numpy(vis_uvs_np)

    # Create a figure to fill out as we go
    if INTERACTIVE:
        n = len(models)
        fig, axes = plt.subplots(2, n, dpi=200, figsize=(2.4 * n, 4.8), squeeze=False)
        plt.ion()
        plt.show()

        black = np.zeros((vis_resolution, vis_resolution, 3), dtype=np.uint8)
        canvases = []
        for i, (model_name, _) in enumerate(models):
            axes[0, i].text(0.5, 1.05, f"{model_name}", horizontalalignment='center', size=8)
            top = axes[0, i].imshow(black, extent=(0, 1, 0, 1), vmin=0, vmax=1)
            bot = axes[1, i].imshow(black, extent=(0, 1, 0, 1), vmin=0, vmax=1)
            canvases.append([top, bot])
            axes[0, i].set_axis_off()
            axes[1, i].set_axis_off()
            fig.tight_layout(h_pad=-1, w_pad=0.5)


    for i, (model_name, model) in enumerate(models):
        print(f"Training model {model_name}")

        assert len(model.parameters()) == 1, "Only one set of parameters is supported in this sample"
        assert model.fan_in == 2 and model.fan_out == 3, "Model must have 2 inputs (UV) and 3 outputs (RGB)"

        ##
        ## Set up optimizer and specialize the slang functions to our model
        ##
        grads = model.gradients()[0]
        parameters = model.parameters()[0]

        parametersF = module.ConvertToFloat(parameters)

        # These match up with the argument names of optimizerStep in texture-training.slang
        optimizer_state = {
            "moments1": spy.NDBuffer.zeros_like(parametersF),
            "moments2": spy.NDBuffer.zeros_like(parametersF),
            "paramF": parametersF,
            "paramH": parameters,
            "grad": grads,
            "learningRate": learning_rate,
            "gradScale": grad_scale
        }
        num_params = parameters.shape[0]

        # Specialize slang functions by substituting generic parameters
        optimizer_step = module.OptimizerStep
        train_texture = module[f"TrainTexture<{model.type_name}, {loss_name} >"]
        eval_model = module[f"EvalModel<{model.type_name} >"]
        eval_loss = module[f"EvalLoss<{loss_name} >"]

        # Begin main training loop
        iteration = 1
        for epoch in range(num_epochs):
            start = time.time()

            cmd = device.create_command_encoder()
            #cmd.begin_compute_pass()
            # Each batch is submitted to a command buffer
            for batch in range(num_batches_per_epoch):
                # Compute gradients
                train_texture.append_to(cmd, model, rng, target_tex, loss_scale)
                # Do one parameter optimization step using those gradients
                optimizer_step.append_to(cmd, idx=spy.call_id((num_params, )), iteration=iteration, **optimizer_state)
                #optimizer_step.append_to(cmd, idx=spy.call_id((num_params, )), iteration=iteration, moments1=optimizer_state["moments1"])
                iteration += 1

            device.submit_command_buffer(cmd.finish())
            device.wait()
            end = time.time()

            # Print out progress info
            elapsed = end - start
            num_samples_per_epoch = math.prod(batch_shape) * num_batches_per_epoch
            progress = (num_samples_per_epoch * (epoch + 1)) // 1000000
            info = (f"Epoch {epoch + 1} complete, "
                    f"{progress}/{sample_target // 1000000} MSamples: "
                    f"Time: {elapsed:.3f}s "
                    f"Throughput: {num_samples_per_epoch / elapsed * 1e-6:.2f} MSamples/s")

            # In the interactive case, draw updates to window and compute loss. This goes
            # through the CPU, so this is quite slow
            if INTERACTIVE:
                current_prediction = eval_model(model, vis_uvs, _result=np.ndarray)
                loss_val = np.mean(eval_loss(vis_uvs, current_prediction, target_tex, _result=np.ndarray))
                diff = module.TextureDifference(vis_uvs, current_prediction, target_tex, 10.0, _result=np.ndarray)

                info += f" Loss: {loss_val:.3f}"

                current_prediction = np.clip(current_prediction, 0, 1)
                diff = np.clip(diff, 0, 1)

                canvases[i][0].set_data(current_prediction)
                canvases[i][1].set_data(diff)
                fig.canvas.draw()
                fig.canvas.flush_events()
            
            print(info)

    print("Training complete!")

    best_model = models[-1][1]

    weight_path = sample.spy_sample_dir / "weights.json"
    print(f"Writing trained weights of best model to {weight_path}")
    param_dict = best_model.serialize()
    open(weight_path, "w").write(json.dumps(param_dict, indent=4))

    print(f"Compiling inference shader...")
    sample.compile_inference_shader(best_model)

    print(f"Running RTXNS inference...")
    if INTERACTIVE:
        plt.close()
    sample.run_sdk_inference(weight_path)

if __name__ == "__main__":
    training_main()
