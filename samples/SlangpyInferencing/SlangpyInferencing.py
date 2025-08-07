# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from app import App
import slangpy as spy
import numpy as np
import json
from pathlib import Path

AssetsPath = Path(__file__).parents[2] / "assets/data"

# Create the app and load the slang module.
app = App(width=512*3+10*2, height=512, title="Inferencing Example", device_type=spy.DeviceType.vulkan)
module = spy.Module.load_from_file(app.device, "SlangpyInferencing_pyslang.slang")

# Load some materials.
image = spy.Tensor.load_from_image(app.device,
                                   AssetsPath / "slangstars.png", linearize=False)

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

        # Convert weights into coopvec layout for training
        desc = app.device.coopvec_create_matrix_desc(self.outputs, self.inputs, self.layout, spy.DataType.float16, 0)
        weight_count = desc.size // 2 # sizeof(half)
        params_np = np.zeros((weight_count, ), dtype=np.float16)
        app.device.coopvec_convert_matrix_host(weights_np, params_np, dst_layout=self.layout)

        self.biases = app.device.create_buffer(struct_size=2, element_count=self.outputs, data=biases_np)
        self.weights = app.device.create_buffer(struct_size=2, element_count=weight_count, data=params_np)

class Network(spy.InstanceList):
    def __init__(self, data: dict):
        super().__init__(module["Network"])

        assert len(data['layers']) == 3

        self.layer0 = NetworkParameters(data['layers'][0])
        self.layer1 = NetworkParameters(data['layers'][1])
        self.layer2 = NetworkParameters(data['layers'][2])


if spy.Feature.cooperative_vector not in module.device.features:
    raise RuntimeError("Device does not support cooperative vector API")

trained_weights = json.load(open(AssetsPath / 'weights.json'))

network = Network(trained_weights)

while app.process_events():

    # Blit tensor to screen.
    offset = 0
    app.blit(image, size=spy.int2(512), offset=spy.int2(offset,0), bilinear=True, tonemap=False)
    offset += 512 + 10
    res = spy.int2(256,256)

    lr_output = spy.Tensor.empty_like(image)
    module.inference(pixel = spy.call_id(), resolution = res, network = network, _result = lr_output)

    # Blit tensor to screen.
    app.blit(lr_output, size=spy.int2(512, 512), offset=spy.int2(offset, 0), bilinear=True, tonemap=False)
    offset += 512 + 10

    # Loss between downsampled output and quarter res rendered output.
    loss_output = spy.Tensor.empty_like(image)
    module.loss(pixel = spy.call_id(),
                  resolution = res,
                  network = network,
                  reference = image,
                  _result = loss_output)

    # Blit tensor to screen.
    app.blit(loss_output, size=spy.int2(512, 512), offset=spy.int2(offset, 0), tonemap=False)
    offset += 512 + 10

    # Present the window.
    app.present()
