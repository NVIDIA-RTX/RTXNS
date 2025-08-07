#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import slangpy as spy
from slangpy.reflection import SlangType
from slangpy.types import NDBuffer
from typing import Any
import numpy as np
import math


# Root interface representing a slang type that implements the 
# IModule interface for trainable CoopVec primitives
class CoopVecModule:
    def __init__(self, element_type: spy.DataType, fan_in: int, fan_out: int):
        super().__init__()

        self.element_type = element_type
        self.elem_name = dtype_name(element_type)
        self.fan_in = fan_in
        self.fan_out = fan_out
    
    # Returns name of the slang type representing this module
    @property
    def type_name(self) -> str:
        raise NotImplementedError()
    
    # Returns a dictionary containing the data for the slang struct
    def get_this(self):
        raise NotImplementedError()
    
    # Returns a list of buffers containing the trainable parameters of this module
    def parameters(self) -> tuple[NDBuffer, ...]:
        raise NotImplementedError()
    
    # Returns a list of buffers containing the gradients of the trainable parameters
    def gradients(self) -> tuple[NDBuffer, ...]:
        raise NotImplementedError()


    # The following methods are used for generating the RTXNS shader that will do
    # inferencing on the trained model

    # Returns the name of the slang type used for inferencing (usually the same as training)    
    @property
    def inference_type_name(self) -> str:
        return self.type_name

    # Returns a braced initializer list that initializes the inferencing struct
    def get_initializer(self) -> str:
        raise NotImplementedError()
    
    # Serializes the trained parameters into a dictionary that can be saved
    # to JSON and loaded by the C++ sample
    def serialize(self) -> dict[str, Any]:
        return {}
    

# Chains multiple modules together into a new module
class ModuleChain(CoopVecModule):
    def __init__(self, *modules: CoopVecModule):
        if len(modules) < 2:
            raise ValueError("Module chain needs at least two modules")
        
        self.modules = modules

        for a, b in zip(modules[:-1], modules[1:]):
            if a.fan_out != b.fan_in:
                raise ValueError(f"Chained modules are incompatible ({a.fan_out} != {b.fan_in})")
            if a.element_type != b.element_type:
                raise ValueError(f"Chained modules are incompatible ({a.element_type} != {b.element_type})")

        super().__init__(modules[0].element_type, modules[0].fan_in, modules[-1].fan_out)

    @property
    def type_name(self) -> SlangType:
        name = self.modules[-1].type_name
        for m in reversed(self.modules[:-1]):
            name = ("rtxns::ModuleChain<"
                f"{self.elem_name}, "
                f"{m.fan_in}, {m.fan_out}, {self.fan_out}, "
                f"{m.type_name}, "
                f"{name}>")
        return name
    
    def get_this(self):
        result = self.modules[-1].get_this()
        for m in reversed(self.modules[:-1]):
            result = {
                "first": m.get_this(),
                "second": result
            }
        return result
    
    def parameters(self) -> tuple[NDBuffer, ...]:
        return sum((m.parameters() for m in self.modules), ())
    
    def gradients(self) -> tuple[NDBuffer, ...]:
        return sum((m.gradients() for m in self.modules), ())

    @property
    def inference_type_name(self) -> SlangType:
        name = self.modules[-1].inference_type_name
        for m in reversed(self.modules[:-1]):
            name = ("rtxns::ModuleChain<"
                f"{self.elem_name}, "
                f"{m.fan_in}, {m.fan_out}, {self.fan_out}, "
                f"{m.inference_type_name}, "
                f"{name}>")
        return name
    
    def get_initializer(self):
        result = self.modules[-1].get_initializer()
        for m in reversed(self.modules[:-1]):
            result = "{" + m.get_initializer() + ", " + result + "}"
        return result
    
    def serialize(self) -> dict[str, Any]:
        result = {}
        for m in self.modules:
            result.update(m.serialize())
        return result


# Frequency encoding that maps each input parameter into a series
# of sines and cosines with increasing frequency
class FrequencyEncoding(CoopVecModule):
    def __init__(self, dtype: spy.DataType, input_width: int, num_octaves: int):
        self.num_octaves = num_octaves
        output_width = num_octaves * input_width * 2
        
        super().__init__(dtype, input_width, output_width)
        
        self._type_name = f"rtxns::FrequencyEncoding<{self.elem_name}, {input_width}, {num_octaves}>"

    @property
    def type_name(self) -> SlangType:
        return self._type_name
    
    def get_this(self):
        return {}
    
    def get_initializer(self):
        return "{}"
    
    def parameters(self) -> tuple[NDBuffer, ...]:
        return ()

    def gradients(self) -> tuple[NDBuffer, ...]:
        return ()
    

# Root class for all activations (i.e. that implement IActivation)
class Activation:
    def __init__(self, act_name: str):
        self.act_name = act_name

    def type_name(self, dtype: spy.DataType, width: int) -> str:
        return f"{self.act_name}<{dtype_name(dtype)}, {width}>"
    
    def get_this(self):
        return {}
    
    def get_initializer(self):
        return "{}"

        
class NoneAct(Activation):
    def __init__(self):
        super().__init__("rtxns::mlp::NoneAct")


class LinearAct(Activation):
    def __init__(self, scale = 1):
        super().__init__("rtxns::mlp::LinearAct")
        self.scale = scale

    def get_this(self):
        return { "a": self.scale }
    
    def get_initializer(self):
        return f"{{{self.scale}}}"


class ExponentialAct(Activation):
    def __init__(self):
        super().__init__("rtxns::mlp::ExponentialAct")


class ShiftedExponentialAct(Activation):
    def __init__(self):
        super().__init__("rtxns::mlp::ShiftedExponentialAct")


class ReLUAct(Activation):
    def __init__(self):
        super().__init__("rtxns::mlp::ReLUAct")


class LeakyReLUAct(Activation):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__("rtxns::mlp::LeakyReLUAct")
        self.negative_slope = negative_slope

    def get_this(self):
        return { "a": self.negative_slope }
    
    def get_initializer(self):
        return f"{{{self.negative_slope}}}"


class SigmoidAct(Activation):
    def __init__(self):
        super().__init__("rtxns::mlp::SigmoidAct")


class SwishAct(Activation):
    def __init__(self):
        super().__init__("rtxns::mlp::SwishAct")


class TanhAct(Activation):
    def __init__(self):
        super().__init__("rtxns::mlp::TanhAct")


# Stores the buffer and offset info that backs of multiple CoopVec matrices
# and biases packed into a single structuredbuffer
class CoopVecParams:
    def __init__(self, device: spy.Device, dtype: spy.DataType, layout: spy.CoopVecMatrixLayout, widths: tuple[int, ...]):
        super().__init__()

        self.dtype = dtype
        self.layout = layout
        self.elem_size = dtype_size(dtype)

        # First, compute offset and size info for the requested layout and layer widths
        self.matrix_descs: list[spy.CoopVecMatrixDesc] = []
        self.bias_offsets = []
        cur_offset = 0
        for fan_in, fan_out in zip(widths[:-1], widths[1:]):
            cur_offset = device.coopvec_align_matrix_offset(cur_offset)
            desc = device.coopvec_create_matrix_desc(fan_out, fan_in, layout, dtype, cur_offset)
            self.matrix_descs.append(desc)
            cur_offset += self.matrix_descs[-1].size

            cur_offset = device.coopvec_align_vector_offset(cur_offset)
            self.bias_offsets.append(cur_offset)
            cur_offset += desc.rows * self.elem_size

        # Create the buffer
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.shared
        self.elem_count = cur_offset // self.elem_size
        self.buffer = NDBuffer(device, dtype_name(dtype), self.elem_count, usage=usage)

        # Create a CPU buffer and pre-slice it into weights and biases that can be
        # more easily handled later
        np_type = dtype_to_numpy(dtype)
        self.np_buffer = self.buffer.to_numpy()
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for b_offset, desc in zip(self.bias_offsets, self.matrix_descs):
            b_index = b_offset // self.elem_size
            m_index = desc.offset // self.elem_size
            np_bias = self.np_buffer[b_index:b_index + desc.rows]
            np_weight = self.np_buffer[m_index:m_index + desc.size // self.elem_size]

            self.biases.append(np_bias)
            self.weights.append(np_weight)
    
    # Copy weights from GPU to CPU
    def to_cpu(self):
        self.np_buffer[:] = self.buffer.to_numpy()
    
    # Copy weights from CPU to GPU
    def to_gpu(self):
        self.buffer.copy_from_numpy(self.np_buffer)

    @property
    def weight_offsets(self):
        return [desc.offset for desc in self.matrix_descs]


# A trainable CoopVec MLP with configurable widths and activations
# The hidden_act activation will be applied to every intermediate output,
# and output_act to the last output
# This stores parameters both in training friendly layout (for fast training)
# and row major layout for easy manipulation. You can copy back and forth
# with to_rowmajor and to_coopvec
class TrainableMLP(CoopVecModule):
    def __init__(self,
                 device: spy.Device,
                 dtype: spy.DataType,
                 num_hidden_layers: int,
                 input_width: int,
                 hidden_width: int,
                 output_width: int,
                 hidden_act: Activation,
                 output_act: Activation):
        
        super().__init__(dtype, input_width, output_width)

        if num_hidden_layers <= 0:
            raise ValueError(f"Must have at least one hidden layer")

        self.device = device
        self.hidden_width = hidden_width
        self.hidden_act = hidden_act
        self.output_act = output_act

        self.widths = \
            (input_width, ) \
            + (hidden_width, ) * num_hidden_layers \
            + (output_width, )
        self.num_layers = len(self.widths) - 1

        # Create wo
        self.params_rowmaj = CoopVecParams(device, dtype, spy.CoopVecMatrixLayout.row_major, self.widths)

        # Initialize weights to random values and copy to row-major params
        rng = np.random.default_rng(seed=12345)
        for i, weight_slice in enumerate(self.params_rowmaj.weights):
            # Xavier uniform initialization
            std = math.sqrt(2.0 / (self.widths[i] + self.widths[i + 1]))
            a = math.sqrt(3.0) * std
            weight_slice[:] = rng.uniform(-a, a, (len(weight_slice), ))
        self.params_rowmaj.to_gpu()

        # Create coopvec parameters and initialize
        layout = spy.CoopVecMatrixLayout.training_optimal
        self.params_coopvec = CoopVecParams(device, dtype, layout, self.widths)

        param_buf = self.params_coopvec.buffer
        self.param_buffer = param_buf
        self.grad_buffer = NDBuffer(self.device, param_buf.dtype, shape=param_buf.shape, usage=param_buf.usage)
        self.grad_buffer.clear()
        self.to_coopvec()


    def _copy(self, src: CoopVecParams, dst: CoopVecParams):
        src.to_cpu()
        dst.to_cpu()
        for i in range(self.num_layers):
            dst.biases[i][:] = src.biases[i]
        dst.to_gpu()
        
        self.device.coopvec_convert_matrix_device(
            src.buffer.storage, src.matrix_descs, dst.buffer.storage, dst.matrix_descs)

    def to_coopvec(self):
        self._copy(self.params_rowmaj, self.params_coopvec)

    def to_row_major(self):
        self._copy(self.params_coopvec, self.params_rowmaj)

    @property
    def type_name(self) -> SlangType:
        return (f"rtxns::TrainableMLPModule<"
            f"{self.elem_name}, "
            f"{self.num_layers - 1}, "
            f"{self.fan_in}, {self.hidden_width}, {self.fan_out}, "
            f"{dtype_to_component_type(self.element_type)}, "
            f"{self.hidden_act.type_name(self.element_type, self.hidden_width)}, "
            f"{self.output_act.type_name(self.element_type, self.fan_out)}>")

    def parameters(self) -> tuple[NDBuffer, ...]:
        return (self.param_buffer, )
    
    def gradients(self) -> tuple[NDBuffer, ...]:
        return (self.grad_buffer, )
    

    def get_this(self):
        return {
            "parameters": self.param_buffer.storage,
            "derivatives": self.grad_buffer.storage,
            "matrixOffsets": self.params_coopvec.weight_offsets,
            "biasOffsets": self.params_coopvec.bias_offsets,
            "hiddenAct": self.hidden_act.get_this(),
            "outputAct": self.output_act.get_this(),
        }

    @property    
    def inference_type_name(self) -> str:
        return (f"rtxns::InferenceMLPModule<"
            f"{self.elem_name}, "
            f"{self.num_layers - 1}, "
            f"{self.fan_in}, {self.hidden_width}, {self.fan_out}, "
            f"{dtype_to_component_type(self.element_type)}, "
            f"{self.hidden_act.type_name(self.element_type, self.hidden_width)}, "
            f"{self.output_act.type_name(self.element_type, self.fan_out)}>")
    
    def get_initializer(self):
        # Assumes variables weights, wo and bo to be in the scope
        hidden = self.hidden_act.get_initializer()
        output = self.output_act.get_initializer()
        w_offsets = ", ".join(f"wo[{i}]" for i in range(self.num_layers))
        b_offsets = ", ".join(f"bo[{i}]" for i in range(self.num_layers))
        return f"{{weights, {{{w_offsets}}}, {{{b_offsets}}}, {hidden}, {output}}}"

    def serialize(self):
        result = {
            'layers': []
        }

        self.to_row_major()
        self.params_rowmaj.to_cpu()

        for i, (w, b) in enumerate(zip(self.params_rowmaj.weights, self.params_rowmaj.biases)):
            result["layers"].append({
                "num_inputs": self.widths[i],
                "num_outputs": self.widths[i + 1],
                "weights": w.flatten().tolist(),
                "biases": b.flatten().tolist()
            })

        return result


def dtype_name(dtype: spy.DataType):
    if dtype == spy.DataType.float16:
        return "half"
    elif dtype == spy.DataType.float32:
        return "float"
    else:
        raise ValueError(f"Unsupported CoopVec datatype '{dtype}'")
    

def dtype_to_numpy(dtype: spy.DataType):
    if dtype == spy.DataType.float16:
        return np.float16
    elif dtype == spy.DataType.float32:
        return np.float32
    else:
        raise ValueError(f"Unsupported CoopVec datatype '{dtype}'")
    

def dtype_to_component_type(dtype: spy.DataType):
    if dtype == spy.DataType.float16:
        return "CoopVecComponentType::Float16"
    elif dtype == spy.DataType.float32:
        return "CoopVecComponentType::Float32"
    else:
        raise ValueError(f"Unsupported CoopVec datatype '{dtype}'")


def dtype_size(dtype: spy.DataType):
    if dtype in (spy.DataType.int8, spy.DataType.uint8):
        return 1
    elif dtype in (spy.DataType.int16, spy.DataType.uint16, spy.DataType.float16):
        return 2
    elif dtype in (spy.DataType.int32, spy.DataType.uint32, spy.DataType.float32):
        return 4
    elif dtype in (spy.DataType.int64, spy.DataType.uint64, spy.DataType.float64):
        return 8
    else:
        raise ValueError(f"Unsupported CoopVec datatype '{dtype}'")
