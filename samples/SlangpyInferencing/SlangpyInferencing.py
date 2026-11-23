# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy as spy
import numpy as np
import json
from pathlib import Path

import matplotlib.pyplot as plt

SampleDir = Path(__file__).parent
AssetsPath = Path(__file__).parents[2] / "assets/data"

# Create a headless device with the include paths needed by the sample shaders.
# Results are read back to the CPU and displayed with matplotlib, so no window
# or swapchain is needed (same approach as the SlangpyTraining sample).
def create_device():
    device = spy.Device(
        type=spy.DeviceType.vulkan,
        compiler_options=spy.SlangCompilerOptions({
            "include_paths": [
                SampleDir,
                Path(spy.__file__).parent / "slang",
            ],
        }),
    )
    print("Selected adapter", device.info.adapter_name)
    return device

device = create_device()
module = spy.Module.load_from_file(device, "SlangpyInferencing_pyslang.slang")

# Load some materials.
image = spy.Tensor.load_from_image(device, AssetsPath / "slangstars.png", linearize=False)

class NetworkParameters(spy.InstanceList):
    def __init__(self, data: dict):
        inputs, outputs = data['num_inputs'], data['num_outputs']
        super().__init__(module[f"NetworkParameters<{inputs},{outputs}>"])

        self.inputs = inputs
        self.outputs = outputs
        self.layout = spy.CoopVecMatrixLayout.inferencing_optimal

        print(outputs, inputs, outputs * inputs, len(data['weights']))

        # Load values of biases and weights
        weights_np = np.array(data['weights'], dtype=np.float16).reshape((outputs, inputs))
        biases_np = np.array(data['biases'], dtype=np.float16)

        # Convert weights into the device's optimal coopvec inferencing layout.
        desc = device.create_coop_vec_matrix_desc(
            self.outputs,
            self.inputs,
            self.layout,
            spy.DataType.float16,
        )
        weight_count = desc.size // 2 # sizeof(half)
        params_np = np.zeros((weight_count, ), dtype=np.float16)
        device.convert_coop_vec_matrix(
            params_np,
            weights_np,
            dst_layout=self.layout,
            src_layout=spy.CoopVecMatrixLayout.row_major,
        )

        self.biases = device.create_buffer(
            struct_size=2,
            element_count=self.outputs,
            usage=spy.BufferUsage.shader_resource,
            data=biases_np,
        )
        self.weights = device.create_buffer(
            struct_size=2,
            element_count=weight_count,
            usage=spy.BufferUsage.shader_resource,
            data=params_np,
        )

class Network(spy.InstanceList):
    def __init__(self, data: dict):
        super().__init__(module["Network"])

        assert len(data['layers']) == 3

        self.layer0 = NetworkParameters(data['layers'][0])
        self.layer1 = NetworkParameters(data['layers'][1])
        self.layer2 = NetworkParameters(data['layers'][2])


if spy.Feature.cooperative_vector not in device.features:
    raise RuntimeError("Device does not support cooperative vector API")

trained_weights = json.load(open(AssetsPath / 'weights.json'))

network = Network(trained_weights)

res = spy.int2(256, 256)

# Run inference for every pixel of the output tensor.
lr_output = spy.Tensor.empty_like(image)
module.inference(pixel = spy.call_id(), resolution = res, network = network, _result = lr_output)

# Loss between the inferred output and the reference image.
loss_output = spy.Tensor.empty_like(image)
module.loss(pixel = spy.call_id(),
              resolution = res,
              network = network,
              reference = image,
              _result = loss_output)

# Read the results back to the CPU and show them side by side:
# the original picture, the inferred picture and the loss image.
panels = [
    ("Reference", image),
    ("Inference", lr_output),
    ("Loss", loss_output),
]

fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.3), num="Inferencing Example")
for ax, (name, tensor) in zip(axes, panels):
    pixels = np.clip(np.abs(tensor.to_numpy()), 0, 1)
    ax.imshow(pixels, interpolation="bilinear")
    ax.set_title(name)
    ax.set_axis_off()
fig.tight_layout()
plt.show()
