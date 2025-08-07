# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import slangpy as spy
from pathlib import Path
from typing import Any, Union
import subprocess
import os

from NeuralModules import CoopVecModule

class SDKSample:
    def __init__(self, args: list[str]):
        super().__init__()

        # Set up directories to find includes and executables
        self.spy_dir = Path(spy.__file__).parent / "slang"
        self.sdk_root = Path(__file__).parent.parent.parent
        self.sdk_data_dir = self.sdk_root / "assets/data"
        self.rtxns_dir = self.sdk_root / "src/NeuralShading_Shaders"
        self.spy_sample_dir = self.sdk_root / "samples/SlangpyTraining"
        self.donut_dir = self.sdk_root / "external/donut/include"
        self.slang_compiler = self.sdk_root / "bin/slangc.bat"

        search_root = self.sdk_root / "bin"
        bin_ext = ".exe" if os.name == "nt" else ""
        inference_candidates = [f for f in search_root.glob(f"**/SlangpyTraining{bin_ext}") if f.is_file()]
        shadermake_candidates = [f for f in search_root.glob(f"**/ShaderMake{bin_ext}") if f.is_file()]
        
        if len(inference_candidates) == 0:
            print(f"Warning: Could not find SlangpyTraining executable within {search_root}. "
                  "C++ sample will not be launched after training.")
            self.inference_sample_path = None
        else:
            self.inference_sample_path = inference_candidates[0]
            if len(inference_candidates) > 1:
                print(f"Warning: Found multiple possible SlangpyTraining executables. Picking {self.inference_sample_path}")
            else:
                print(f"Found SlangpyTraining executable at {self.inference_sample_path}")

        if len(shadermake_candidates) == 0:
            print(f"Warning: Could not find ShaderMake executable within {search_root}. "
                  "C++ sample will not be launched after training.")
            self.shadermake_path = None
        else:
            self.shadermake_path = shadermake_candidates[0]
            if len(shadermake_candidates) > 1:
                print(f"Warning: Found multiple possible ShaderMake executables. Picking {self.shadermake_path}")
            else:
                print(f"Found ShaderMake executable at {self.shadermake_path}")

        self.include_dirs = [
            self.rtxns_dir,
            self.spy_dir,
            self.spy_sample_dir
        ]

        for field in ("spy_dir", "sdk_root", "sdk_data_dir", "rtxns_dir", "spy_sample_dir", "donut_dir", "slang_compiler"):
            path: Path = getattr(self, field)
            if not path.exists():
                print(f"Warning: Can't find path {field} at {path}. This may cause errors.")
        
        self.device = self._create_device()
    
    # Create an sgl device and setup default include directories
    def _create_device(self):
        device = spy.Device(
            type=spy.DeviceType.vulkan,
            compiler_options=spy.SlangCompilerOptions({
                "include_paths": self.include_dirs,
                "disable_warnings": [
                    "41018", # Overzealous uninitialized-out-parameter warning
                    "41012"  # Coop vec capability warning
                ]
            }),
        )

        print("Selected adapter", device.info.adapter_name)

        return device
    
    def load_texture(self, path: Union[str,Path]):
        bmp = spy.Bitmap(self.sdk_data_dir / path)
        loader = spy.TextureLoader(self.device)
        target_tex = loader.load_texture(bmp, {"load_as_normalized": True})
        return target_tex
    
    # Take a trained model and distill it to defines and compile it
    def compile_inference_shader(self, model: CoopVecModule):
        if self.inference_sample_path is None or self.shadermake_path is None:
            print("Missing executables, skipping compilation.")
            return
        
        if len(model.parameters()) > 1:
            raise ValueError("Shader generation only supports a single parameter buffer")

        defines = [
            ("MODEL_TYPE", f'"{model.inference_type_name}"'),
            ("MODEL_INITIALIZER", f'"{model.get_initializer()}"'),
            ("VECTOR_FORMAT", model.elem_name),
        ]

        self.compile_shader("SlangpyInference.slang", defines)

    def compile_shader(self, shader_path: str, defines: list[Union[str,tuple[str, Any]]]):
        config_path = self.spy_sample_dir / "trained_shaders.cfg"
        with open(config_path, "w") as file:
            file.write(f"{shader_path} -E main_cs -T cs")

        output_path = self.inference_sample_path.parent / "shaders/SlangpyTraining/spirv"
        
        args = [
            self.shadermake_path,
            "--config", config_path,
            "-o", output_path,
            "--compiler", self.slang_compiler,
            "--platform", "SPIRV",
            "--flatten",
            "--binaryBlob",
            "--outputExt", ".bin",
            "--slang",
            "--tRegShift", "0",
            "--sRegShift", "128",
            "--bRegShift", "256",
            "--uRegShift", "384",
            "--vulkanVersion", "1.2",
            "--matrixRowMajor",
            "--force",
            "-X", "-capability spvCooperativeVectorNV -capability spvCooperativeVectorTrainingNV",
        ]
        for d in defines + ["SPIRV", "TARGET_VULKAN"]:
            if isinstance(d, str):
                args.extend(("-D", d))
            else:
                args.extend(("-D", f"{d[0]}={d[1]}"))
        
        for include_dir in self.include_dirs + [self.donut_dir]:
            args.extend(("-I", include_dir))

        result = subprocess.run(args, text=True, capture_output=True)
        if result.stderr:
            raise RuntimeError(f"ShaderMake exited with errors: {result.stderr}")
        stdout = str(result.stdout)
        if stdout.find(": error") != -1:
            raise RuntimeError(f"slang compiler exited with errors: {stdout}")
    
    def run_sdk_inference(self, model_weights: Path):
        if self.inference_sample_path is None or self.shadermake_path is None:
            print("Missing executables, skipping C++ sample.")
            return
        
        subprocess.run([self.inference_sample_path, model_weights])
