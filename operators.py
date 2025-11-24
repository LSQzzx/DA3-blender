import bpy
from pathlib import Path
import os
import torch
import numpy as np
from .utils import run_model, import_point_cloud, create_cameras

add_on_path = Path(__file__).parent
MODELS_DIR = os.path.join(add_on_path, 'models')
_URLS = {
    'da3-small': "https://huggingface.co/depth-anything/DA3-SMALL/resolve/main/model.safetensors",
    'da3-base': "https://huggingface.co/depth-anything/DA3-BASE/resolve/main/model.safetensors",
    'da3-large': "https://huggingface.co/depth-anything/DA3-LARGE/resolve/main/model.safetensors",
    'da3-giant': "https://huggingface.co/depth-anything/DA3-GIANT/resolve/main/model.safetensors",
}
model = None
current_model_name = None

def get_model_path(model_name):
    return os.path.join(MODELS_DIR, f'{model_name}.safetensors')

def get_model(model_name):
    global model, current_model_name
    if model is None or current_model_name != model_name:
        from depth_anything_3.api import DepthAnything3
        model = DepthAnything3(model_name=model_name)
        model_path = get_model_path(model_name)
        if os.path.exists(model_path):
            from safetensors.torch import load_file
            weight = load_file(model_path)
            model.load_state_dict(weight, strict=False)
        else:
            raise FileNotFoundError(f"Model file {model_name} not found. Please download it first.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        current_model_name = model_name
    return model

class DownloadModelOperator(bpy.types.Operator):
    bl_idname = "da3.download_model"
    bl_label = "Download DA3 Model"

    def execute(self, context):
        model_name = context.scene.da3_model_name
        model_path = get_model_path(model_name)
        
        if os.path.exists(model_path):
            self.report({'INFO'}, f"Model {model_name} already downloaded.")
            return {'FINISHED'}
        
        if model_name not in _URLS:
            self.report({'ERROR'}, f"Unknown model: {model_name}")
            return {'CANCELLED'}
            
        try:
            print(f"Downloading model {model_name}...")
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.hub.download_url_to_file(_URLS[model_name], model_path)
            self.report({'INFO'}, f"Model {model_name} downloaded successfully.")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to download model {model_name}: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}

    @classmethod
    def poll(cls, context):
        model_name = context.scene.da3_model_name
        model_path = get_model_path(model_name)
        return not os.path.exists(model_path)


class GeneratePointCloudOperator(bpy.types.Operator):
    bl_idname = "da3.generate_point_cloud"
    bl_label = "Generate Point Cloud"

    def execute(self, context):
        input_folder = context.scene.da3_input_folder
        model_name = context.scene.da3_model_name
        
        if not input_folder or not os.path.isdir(input_folder):
            self.report({'ERROR'}, "Please select a valid input folder.")
            return {'CANCELLED'}
        try:
            model = get_model(model_name)
            predictions = run_model(input_folder, model)
            import_point_cloud(predictions)
            self.report({'INFO'}, "Point cloud generated and imported successfully.")
            create_cameras(predictions)
            self.report({'INFO'}, "Cameras generated successfully.")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to generate point cloud: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}

    @classmethod
    def poll(cls, context):
        model_name = context.scene.da3_model_name
        model_path = get_model_path(model_name)
        return os.path.exists(model_path) and context.scene.da3_input_folder != ""