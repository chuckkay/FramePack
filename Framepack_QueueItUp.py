from diffusers_helper.hf_login import login
import os
import re
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
import json
import traceback
from dataclasses import dataclass, asdict
from typing import Optional
import uuid
import configparser
import ast
import random
import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
# from diffusers_helper.load_lora import load_lora
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
import shutil
import cv2

# After imports, before other functions
def is_model_downloaded(model_value):
    """Check if a model is downloaded by checking for the LOCAL- prefix"""
    if isinstance(model_value, str):
        return model_value.startswith('LOCAL-')
    return False

def set_model_as_default(model_type, model_value):
    """Set a specific model as the default in settings.ini"""
    try:
        # First check if model is downloaded (has LOCAL- prefix)
        if not model_value.startswith('LOCAL-'):
            return f"Cannot set {model_type} as default - model must be downloaded first", None, None, None, None, None, None, None, None
            
        config = configparser.ConfigParser()
        config.read(INI_FILE)
        
        if 'Model Defaults' not in config:
            config['Model Defaults'] = {}
            
        # Remove the display prefix to get actual folder name
        actual_model = Config.model_name_mapping.get(model_value, model_value.replace('LOCAL-', '').replace(' - CURRENT DEFAULT MODEL', ''))
        
        # Update the config
        config['Model Defaults'][f'DEFAULT_{model_type.upper()}'] = repr(actual_model)
        
        # Save to file
        with open(INI_FILE, 'w') as f:
            config.write(f)
            
        # Update Config object
        setattr(Config, f'DEFAULT_{model_type.upper()}', actual_model)
        
        # Get updated model lists
        models = get_available_models(include_online=False)
        
        # Return status message and individual model lists
        return (
            f"{model_type} set as default successfully. Restart required for changes to take effect.",
            models['transformer'],
            models['text_encoder'],
            models['text_encoder_2'],
            models['tokenizer'],
            models['tokenizer_2'],
            models['vae'],
            models['feature_extractor'],
            models['image_encoder']
        )
    except Exception as e:
        return f"Error setting {model_type} as default: {str(e)}", None, None, None, None, None, None, None, None

def set_all_models_as_default(transformer, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder):
    """Set all models as default at once"""
    try:
        # Check if all models are downloaded
        models = [transformer, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder]
        model_types = ['transformer', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'vae', 'feature_extractor', 'image_encoder']
        
        for model, model_type in zip(models, model_types):
            if not model.startswith('LOCAL-'):
                return f"Cannot set {model_type} as default - model must be downloaded first", None, None, None, None, None, None, None, None
        
        config = configparser.ConfigParser()
        config.read(INI_FILE)
        
        if 'Model Defaults' not in config:
            config['Model Defaults'] = {}
            
        # Update all models
        success_messages = []
        for model, model_type in zip(models, model_types):
            actual_model = Config.model_name_mapping.get(model, model.replace('LOCAL-', '').replace(' - CURRENT DEFAULT MODEL', ''))
            config['Model Defaults'][f'DEFAULT_{model_type.upper()}'] = repr(actual_model)
            setattr(Config, f'DEFAULT_{model_type.upper()}', actual_model)
            success_messages.append(f"{model_type}: {actual_model}")
            
        # Save to file
        with open(INI_FILE, 'w') as f:
            config.write(f)
            
        # Get updated model lists
        models = get_available_models(include_online=False)
        
        return (
            "All models set as default successfully:\n" + "\n".join(success_messages) + "\n\nRestart required for changes to take effect.",
            models['transformer'],
            models['text_encoder'],
            models['text_encoder_2'],
            models['tokenizer'],
            models['tokenizer_2'],
            models['vae'],
            models['feature_extractor'],
            models['image_encoder']
        )
    except Exception as e:
        return f"Error setting models as default: {str(e)}", None, None, None, None, None, None, None, None

def restore_model_default(model_type):
    """Restore a specific model to its original default in settings.ini"""
    try:
        # Get original defaults
        original_defaults = Config.get_original_defaults()
        original_value = original_defaults[f'DEFAULT_{model_type.upper()}']
        
        config = configparser.ConfigParser()
        config.read(INI_FILE)
        
        if 'Model Defaults' not in config:
            config['Model Defaults'] = {}
            
        # Update the config
        config['Model Defaults'][f'DEFAULT_{model_type.upper()}'] = repr(original_value)
        
        # Save to file
        with open(INI_FILE, 'w') as f:
            config.write(f)
            
        # Update Config object
        setattr(Config, f'DEFAULT_{model_type.upper()}', original_value)
        
        # Get updated model lists
        models = get_available_models(include_online=False)
        
        # Return status message and individual model lists
        return (
            f"{model_type} restored to original default successfully. Restart required for changes to take effect.",
            models['transformer'],
            models['text_encoder'],
            models['text_encoder_2'],
            models['tokenizer'],
            models['tokenizer_2'],
            models['vae'],
            models['feature_extractor'],
            models['image_encoder']
        )
    except Exception as e:
        return f"Error restoring {model_type} default: {str(e)}", None, None, None, None, None, None, None, None


def download_model_from_huggingface(model_id):
    """Download a model from Hugging Face and return its local path"""
    try:
        # Handle our display name format
        if model_id.startswith('DOWNLOADED-MODEL-'):
            if hasattr(Config, 'model_name_mapping') and model_id in Config.model_name_mapping:
                model_id = Config.model_name_mapping[model_id]
        
        # Convert from org/model to models--org--model format
        if '/' in model_id:
            org, model = model_id.split('/')
            local_name = f"models--{org}--{model}"
        else:
            local_name = model_id
            
        hub_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_download", "hub")
        model_path = os.path.join(hub_dir, local_name)
        
        # Get token from Config and set in environment
        token = Config._instance.HF_TOKEN if Config._instance else Config.HF_TOKEN
        if not token or token == "add token here":
            alert_print("No valid Hugging Face token found. Please add your token in the settings.")
            return None
            
        # Set token in environment
        os.environ['HF_TOKEN'] = token
        
        if not os.path.exists(model_path):
            debug_print(f"Downloading model {model_id}...")
            try:
                # First check if we can access the model
                from huggingface_hub import HfApi
                api = HfApi()
                try:
                    # Try to get model info first
                    model_info = api.model_info(model_id, token=token)
                    debug_print(f"Model info: {model_info.modelId} - {model_info.tags if hasattr(model_info, 'tags') else 'No tags'}")
                    if hasattr(model_info, 'private') and model_info.private:
                        alert_print(f"Model {model_id} is private and cannot be accessed")
                        return None
                except Exception as e:
                    alert_print(f"Cannot access model {model_id}: {str(e)}")
                    return None
                
                # Use the appropriate model class based on the model type
                if "hunyuanvideo" in model_id.lower():
                    try:
                        AutoencoderKLHunyuanVideo.from_pretrained(
                            model_id, 
                            cache_dir=hub_dir, 
                            use_auth_token=token,
                            local_files_only=False,
                            resume_download=True
                        )
                    except Exception as e:
                        alert_print(f"Error downloading hunyuanvideo model: {str(e)}")
                        if os.path.exists(model_path):
                            import shutil
                            shutil.rmtree(model_path, ignore_errors=True)
                        return None
                elif "flux_redux" in model_id.lower():
                    try:
                        SiglipImageProcessor.from_pretrained(
                            model_id, 
                            cache_dir=hub_dir, 
                            use_auth_token=token,
                            local_files_only=False,
                            resume_download=True
                        )
                        SiglipVisionModel.from_pretrained(
                            model_id, 
                            cache_dir=hub_dir, 
                            use_auth_token=token,
                            local_files_only=False,
                            resume_download=True
                        )
                    except Exception as e:
                        alert_print(f"Error downloading flux_redux model: {str(e)}")
                        if os.path.exists(model_path):
                            import shutil
                            shutil.rmtree(model_path, ignore_errors=True)
                        return None
                elif "framepack" in model_id.lower():
                    try:
                        HunyuanVideoTransformer3DModelPacked.from_pretrained(
                            model_id, 
                            cache_dir=hub_dir, 
                            use_auth_token=token,
                            local_files_only=False,
                            resume_download=True
                        )
                    except Exception as e:
                        alert_print(f"Error downloading framepack model: {str(e)}")
                        if os.path.exists(model_path):
                            import shutil
                            shutil.rmtree(model_path, ignore_errors=True)
                        return None
                debug_print(f"Model downloaded to {model_path}")
            except Exception as e:
                alert_print(f"Error during download process: {str(e)}")
                if os.path.exists(model_path):
                    import shutil
                    shutil.rmtree(model_path, ignore_errors=True)
                return None
                
        # Return display name if we have it
        display_name = next((k for k, v in Config.model_name_mapping.items() if v == local_name), None) if hasattr(Config, 'model_name_mapping') else None
        return display_name if display_name else local_name
        
    except Exception as e:
        alert_print(f"Error in download process: {str(e)}")
        traceback.print_exc()
        return None

def get_available_models(include_online=False):
    """Get list of available models from hub directory and optionally from Hugging Face"""
    debug_print(f"Starting get_available_models with include_online={include_online}")
    hub_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_download", "hub")
    debug_print(f"Hub directory: {hub_dir}")
    
    # Dictionary to store the mapping between display names and actual folder names
    name_mapping = {}
    
    models = {
        'text_encoder': [],
        'text_encoder_2': [],
        'tokenizer': [],
        'tokenizer_2': [],
        'vae': [],
        'feature_extractor': [],
        'image_encoder': [],
        'transformer': []
    }
    
    # Get current defaults from Config
    user_defaults = {
        'transformer': Config.DEFAULT_TRANSFORMER,
        'text_encoder': Config.DEFAULT_TEXT_ENCODER,
        'text_encoder_2': Config.DEFAULT_TEXT_ENCODER_2,
        'tokenizer': Config.DEFAULT_TOKENIZER,
        'tokenizer_2': Config.DEFAULT_TOKENIZER_2,
        'vae': Config.DEFAULT_VAE,
        'feature_extractor': Config.DEFAULT_FEATURE_EXTRACTOR,
        'image_encoder': Config.DEFAULT_IMAGE_ENCODER
    }
    
    # Get original defaults
    original_defaults = Config.get_original_defaults()
    
    # Get local models
    if os.path.exists(hub_dir):
        debug_print("Scanning local models...")
        for item in os.listdir(hub_dir):
            if os.path.isdir(os.path.join(hub_dir, item)):
                debug_print(f"Found local model directory: {item}")
                # Create base display name with prefix
                display_name = f"LOCAL-{item}"
                # Store mapping
                name_mapping[display_name] = item
                
                # Map models to their correct categories using display name
                if "hunyuanvideo" in item.lower():
                    # Function to handle model categorization
                    def add_model_with_suffix(model_type):
                        is_user_default = item == user_defaults[model_type]
                        is_original_default = item == original_defaults[f'DEFAULT_{model_type.upper()}']
                        
                        if is_original_default and not is_user_default:
                            display_name_suffix = f"{display_name} - ORIGINAL DEFAULT MODEL"
                            name_mapping[display_name_suffix] = item
                            models[model_type].append(display_name_suffix)
                        elif is_user_default:
                            display_name_suffix = f"{display_name} - CURRENT DEFAULT MODEL"
                            name_mapping[display_name_suffix] = item
                            models[model_type].append(display_name_suffix)
                        else:
                            models[model_type].append(display_name)
                    
                    add_model_with_suffix('text_encoder')
                    add_model_with_suffix('text_encoder_2')
                    add_model_with_suffix('tokenizer')
                    add_model_with_suffix('tokenizer_2')
                    add_model_with_suffix('vae')
                        
                elif "flux_redux" in item.lower():
                    def add_model_with_suffix(model_type):
                        is_user_default = item == user_defaults[model_type]
                        is_original_default = item == original_defaults[f'DEFAULT_{model_type.upper()}']
                        
                        if is_original_default and not is_user_default:
                            display_name_suffix = f"{display_name} - ORIGINAL DEFAULT MODEL"
                            name_mapping[display_name_suffix] = item
                            models[model_type].append(display_name_suffix)
                        elif is_user_default:
                            display_name_suffix = f"{display_name} - CURRENT DEFAULT MODEL"
                            name_mapping[display_name_suffix] = item
                            models[model_type].append(display_name_suffix)
                        else:
                            models[model_type].append(display_name)
                    
                    add_model_with_suffix('feature_extractor')
                    add_model_with_suffix('image_encoder')
                        
                elif "framepack" in item.lower():
                    is_user_default = item == user_defaults['transformer']
                    is_original_default = item == original_defaults['DEFAULT_TRANSFORMER']
                    
                    if is_original_default and not is_user_default:
                        display_name_suffix = f"{display_name} - ORIGINAL DEFAULT MODEL"
                        name_mapping[display_name_suffix] = item
                        models['transformer'].append(display_name_suffix)
                    elif is_user_default:
                        display_name_suffix = f"{display_name} - CURRENT DEFAULT MODEL"
                        name_mapping[display_name_suffix] = item
                        models['transformer'].append(display_name_suffix)
                    else:
                        models['transformer'].append(display_name)
    
    debug_print("Local models found:")
    for key, value in models.items():
        debug_print(f"  {key}: {value}")
    
    # Add online models if requested
    if include_online:
        debug_print("Online models requested, starting online search...")
        try:
            from huggingface_hub import HfApi
            
            # Get token from Config and set in environment
            token = Config._instance.HF_TOKEN if Config._instance else Config.HF_TOKEN
            debug_print(f"Token status: {'Valid token found' if token and token != 'add token here' else 'No valid token'}")
            
            if token and token != "add token here":
                try:
                    # Set token in environment for the helper module
                    os.environ['HF_TOKEN'] = token
                    
                    debug_print("Attempting Hugging Face login...")
                    # Use the imported login function from diffusers_helper
                    login(token)
                    debug_print("Login successful, searching for models...")
                    
                    api = HfApi()
                    
                    # Convert generators to lists before processing
                    debug_print("Searching for hunyuanvideo models...")
                    hunyuan_models = list(api.list_models(search="hunyuanvideo", token=token))
                    debug_print(f"Found {len(hunyuan_models)} hunyuan models")
                    
                    debug_print("Searching for flux_redux models...")
                    flux_models = list(api.list_models(search="flux_redux", token=token))
                    debug_print(f"Found {len(flux_models)} flux models")
                    
                    debug_print("Searching for framepack models...")
                    framepack_models = list(api.list_models(search="framepack", token=token))
                    debug_print(f"Found {len(framepack_models)} framepack models")
                    
                    # Add online models to the lists
                    for model in hunyuan_models:
                        model_id = model.id
                        debug_print(f"Processing hunyuan model: {model_id}")
                        
                        # Check if this is a default model
                        is_user_default = (model_id == convert_model_path(user_defaults['text_encoder']) or
                                         model_id == convert_model_path(user_defaults['text_encoder_2']) or
                                         model_id == convert_model_path(user_defaults['tokenizer']) or
                                         model_id == convert_model_path(user_defaults['tokenizer_2']) or
                                         model_id == convert_model_path(user_defaults['vae']))
                                         
                        is_original_default = (model_id == convert_model_path(original_defaults['DEFAULT_TEXT_ENCODER']) or
                                             model_id == convert_model_path(original_defaults['DEFAULT_TEXT_ENCODER_2']) or
                                             model_id == convert_model_path(original_defaults['DEFAULT_TOKENIZER']) or
                                             model_id == convert_model_path(original_defaults['DEFAULT_TOKENIZER_2']) or
                                             model_id == convert_model_path(original_defaults['DEFAULT_VAE']))
                        
                        display_id = model_id
                        if is_original_default and not is_user_default:
                            display_id = f"{model_id} - ORIGINAL DEFAULT MODEL"
                        elif is_user_default:
                            display_id = f"{model_id} - CURRENT DEFAULT MODEL"
                            
                        if display_id not in models['text_encoder']:
                            models['text_encoder'].append(display_id)
                            models['text_encoder_2'].append(display_id)
                            models['tokenizer'].append(display_id)
                            models['tokenizer_2'].append(display_id)
                            models['vae'].append(display_id)
                            
                    for model in flux_models:
                        model_id = model.id
                        debug_print(f"Processing flux model: {model_id}")
                        
                        # Check if this is a default model
                        is_user_default = (model_id == convert_model_path(user_defaults['feature_extractor']) or
                                         model_id == convert_model_path(user_defaults['image_encoder']))
                                         
                        is_original_default = (model_id == convert_model_path(original_defaults['DEFAULT_FEATURE_EXTRACTOR']) or
                                             model_id == convert_model_path(original_defaults['DEFAULT_IMAGE_ENCODER']))
                        
                        display_id = model_id
                        if is_original_default and not is_user_default:
                            display_id = f"{model_id} - ORIGINAL DEFAULT MODEL"
                        elif is_user_default:
                            display_id = f"{model_id} - CURRENT DEFAULT MODEL"
                            
                        if display_id not in models['feature_extractor']:
                            models['feature_extractor'].append(display_id)
                            models['image_encoder'].append(display_id)
                            
                    for model in framepack_models:
                        model_id = model.id
                        debug_print(f"Processing framepack model: {model_id}")
                        
                        # Check if this is a default model
                        is_user_default = model_id == convert_model_path(user_defaults['transformer'])
                        is_original_default = model_id == convert_model_path(original_defaults['DEFAULT_TRANSFORMER'])
                        
                        display_id = model_id
                        if is_original_default and not is_user_default:
                            display_id = f"{model_id} - ORIGINAL DEFAULT MODEL"
                        elif is_user_default:
                            display_id = f"{model_id} - CURRENT DEFAULT MODEL"
                            
                        if display_id not in models['transformer']:
                            models['transformer'].append(display_id)
                            
                except Exception as e:
                    alert_print(f"Error logging in to Hugging Face: {str(e)}")
                    debug_print(f"Login error details: {traceback.format_exc()}")
            else:
                alert_print("No valid Hugging Face token found. Please add your token in the settings.")
        except Exception as e:
            alert_print(f"Error fetching online models: {str(e)}")
            debug_print(f"Online fetch error details: {traceback.format_exc()}")
    
    debug_print("Returning final model lists")
    # Store the name mapping in a global variable or Config
    Config.model_name_mapping = name_mapping
    return models

# Path to settings file
INI_FILE = os.path.join(os.getcwd(), 'settings.ini')

# Path to the quick prompts JSON file
QUICK_LIST_FILE = os.path.join(os.getcwd(), 'quick_prompts.json')

# Queue file path
QUEUE_JSON_FILE = os.path.join(os.getcwd(), 'job_queue.json')

# Temp directory for queue images
temp_queue_images = os.path.join(os.getcwd(), 'temp_queue_images')
os.makedirs(temp_queue_images, exist_ok=True)


# ANSI color codes
YELLOW = '\033[93m'
RED = '\033[31m'
GREEN = '\033[92m'
RESET = '\033[0m'
try:
# Try to load arial font, fall back to default if not available
    font = ImageFont.truetype("arial.ttf", 16)
    small_font = ImageFont.truetype("arial.ttf", 12)
except:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()


def debug_print(message):
    """Print debug messages in yellow color"""
    if debug_mode:
        print(f"{YELLOW}[DEBUG] {message}{RESET}")
    
def alert_print(message):
    """Print alert messages in red color"""
    print(f"{RED}[ALERT] {message}{RESET}")

def info_print(message):
    """Print info messages in green color"""
    print(f"{GREEN}[INFO] {message}{RESET}")


def save_settings(config):
    """Save settings to settings.ini"""
    with open(INI_FILE, 'w') as f:
        config.write(f)

def save_job_defaults_from_ui(prompt, n_prompt, use_teacache, seed, job_name, video_length, steps, cfg, gs, rs, gpu_memory, mp4_crf, keep_temp_png, keep_temp_json):
    """Save Job Defaults from UI settings"""
    config = load_settings()
    
    # Ensure sections exist
    if 'Job Defaults' not in config:
        config['Job Defaults'] = {}
    if 'Model Defaults' not in config:
        config['Model Defaults'] = {}
    
    # Save Job Defaults with consistent casing - excluding Model Defaults
    section = config['Job Defaults']
    section['DEFAULT_PROMPT'] = repr(prompt)
    section['DEFAULT_N_PROMPT'] = repr(n_prompt)
    section['DEFAULT_USE_TEACACHE'] = repr(use_teacache)
    section['DEFAULT_SEED'] = repr(seed)
    section['DEFAULT_JOB_NAME'] = repr(job_name)
    section['DEFAULT_VIDEO_LENGTH'] = repr(video_length)
    section['DEFAULT_STEPS'] = repr(steps)
    section['DEFAULT_CFG'] = repr(cfg)
    section['DEFAULT_GS'] = repr(gs)
    section['DEFAULT_RS'] = repr(rs)
    section['DEFAULT_GPU_MEMORY'] = repr(gpu_memory)
    section['DEFAULT_MP4_CRF'] = repr(mp4_crf)
    section['DEFAULT_KEEP_TEMP_PNG'] = repr(keep_temp_png)
    section['DEFAULT_KEEP_TEMP_JSON'] = repr(keep_temp_json)

    
    # Update Config instance with new job defaults
    Config.DEFAULT_PROMPT = prompt
    Config.DEFAULT_N_PROMPT = n_prompt
    Config.DEFAULT_USE_TEACACHE = use_teacache
    Config.DEFAULT_SEED = seed
    Config.DEFAULT_JOB_NAME = job_name
    Config.DEFAULT_VIDEO_LENGTH = video_length
    Config.DEFAULT_STEPS = steps
    Config.DEFAULT_CFG = cfg
    Config.DEFAULT_GS = gs
    Config.DEFAULT_RS = rs
    Config.DEFAULT_GPU_MEMORY = gpu_memory
    Config.DEFAULT_MP4_CRF = mp4_crf
    Config.DEFAULT_KEEP_TEMP_PNG = keep_temp_png
    Config.DEFAULT_KEEP_TEMP_JSON = keep_temp_json
    
    save_settings(config)
    debug_print(f"Saved video_length as {section['DEFAULT_VIDEO_LENGTH']}")
    return "Settings saved successfully!"

@dataclass
class Config:
    """Centralized configuration for default values"""
    _instance = None
    
    # Default prompt settings
    DEFAULT_PROMPT: str = None
    DEFAULT_N_PROMPT: str = None
    DEFAULT_JOB_NAME: str = None
    DEFAULT_VIDEO_LENGTH: float = None
    DEFAULT_GS: float = None
    DEFAULT_STEPS: int = None
    DEFAULT_USE_TEACACHE: bool = None
    DEFAULT_SEED: int = None
    DEFAULT_CFG: float = None
    DEFAULT_RS: float = None
    DEFAULT_GPU_MEMORY: float = None
    DEFAULT_MP4_CRF: int = None
    DEFAULT_KEEP_TEMP_PNG: bool = None
    DEFAULT_KEEP_TEMP_JSON: bool = None
    DEFAULT_LATENT_WINDOW_SIZE: int = None  # Added latent window size

    # Model Defaults
    DEFAULT_TRANSFORMER: str = None
    DEFAULT_TEXT_ENCODER: str = None
    DEFAULT_TEXT_ENCODER_2: str = None
    DEFAULT_TOKENIZER: str = None
    DEFAULT_TOKENIZER_2: str = None
    DEFAULT_VAE: str = None
    DEFAULT_FEATURE_EXTRACTOR: str = None
    DEFAULT_IMAGE_ENCODER: str = None

    # System defaults
    OUTPUTS_FOLDER: str = None
    JOB_HISTORY_FOLDER: str = None
    DEBUG_MODE: bool = None
    KEEP_TEMP_MP4: bool = None
    KEEP_COMPLETED_JOB: bool = None
    HF_TOKEN: str = None

    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def save_to_ini(self):
        """Save current config to settings.ini"""
        config = configparser.ConfigParser()
        
        # Ensure sections exist
        if 'Job Defaults' not in config:
            config['Job Defaults'] = {}
        if 'System Defaults' not in config:
            config['System Defaults'] = {}
            
        # Update HF_TOKEN in System Defaults
        config['System Defaults']['hf_token'] = str(self.HF_TOKEN)
        
        # Save to file
        with open(INI_FILE, 'r') as f:
            config.read_file(f)
            
        with open(INI_FILE, 'w') as f:
            config.write(f)

    @classmethod
    def get_original_defaults(cls):
        """Returns a dictionary of original default values - this is the single source of truth for defaults"""
        return {
            'DEFAULT_PROMPT': "The girl dances gracefully, with clear movements, full of charm.",
            'DEFAULT_N_PROMPT': "",
            'DEFAULT_JOB_NAME': "Job-",
            'DEFAULT_VIDEO_LENGTH': 5.0,
            'DEFAULT_GS': 10.0,
            'DEFAULT_STEPS': 25,
            'DEFAULT_USE_TEACACHE': True,
            'DEFAULT_SEED': -1,
            'DEFAULT_CFG': 1.0,
            'DEFAULT_RS': 0.0,
            'DEFAULT_GPU_MEMORY': 6.0,
            'DEFAULT_MP4_CRF': 16,
            'DEFAULT_KEEP_TEMP_PNG': True,
            'DEFAULT_KEEP_TEMP_JSON': True,
            'DEFAULT_LATENT_WINDOW_SIZE': 9,  # Added default latent window size
            'DEFAULT_TRANSFORMER': "models--lllyasviel--FramePack_F1_I2V_HY_20250503",
            'DEFAULT_TEXT_ENCODER': "models--hunyuanvideo-community--HunyuanVideo",
            'DEFAULT_TEXT_ENCODER_2': "models--hunyuanvideo-community--HunyuanVideo",
            'DEFAULT_TOKENIZER': "models--hunyuanvideo-community--HunyuanVideo",
            'DEFAULT_TOKENIZER_2': "models--hunyuanvideo-community--HunyuanVideo",
            'DEFAULT_VAE': "models--hunyuanvideo-community--HunyuanVideo",
            'DEFAULT_FEATURE_EXTRACTOR': "models--lllyasviel--flux_redux_bfl",
            'DEFAULT_IMAGE_ENCODER': "models--lllyasviel--flux_redux_bfl",
            'OUTPUTS_FOLDER': './outputs/',
            'JOB_HISTORY_FOLDER': './job_history/',
            'DEBUG_MODE': False,
            'KEEP_TEMP_MP4': False,
            'KEEP_COMPLETED_JOB': True,
            'HF_TOKEN': 'add token here'
        }

    @classmethod
    def from_settings(cls, config):
        """Create Config instance from settings.ini values"""
        instance = cls()
        
        # Load Job Defaults section
        section = config['Job Defaults']
        section_keys = {k.upper(): k for k in section.keys()}
        
        # Load all non-model values from Job Defaults
        for key, default_value in instance.get_original_defaults().items():
            if key.startswith('DEFAULT_') and not any(model_type in key.lower() for model_type in ['transformer', 'text_encoder', 'tokenizer', 'vae', 'feature_extractor', 'image_encoder']):
                try:
                    # Look up the actual key in a case-insensitive way
                    actual_key = section_keys.get(key.upper(), key)
                    if actual_key not in section:
                        actual_key = section_keys.get(key.lower(), key)
                    
                    value = section.get(actual_key, str(default_value))
                    
                    # Handle different types appropriately
                    if isinstance(default_value, bool):
                        parsed_value = str(value).lower() in ('true', 't', 'yes', 'y', '1')
                    elif isinstance(default_value, int):
                        parsed_value = int(float(str(value).strip("'")))
                    elif isinstance(default_value, float):
                        parsed_value = float(str(value).strip("'"))
                    else:
                        parsed_value = str(value).strip("'")
                    
                    setattr(instance, key, parsed_value)
                except Exception as e:
                    debug_print(f"Error loading {key}: {str(e)}, using default value: {default_value}")
                    setattr(instance, key, default_value)
                    section[key] = str(default_value)
                    save_settings(config)

        # Load Model Defaults section
        if 'Model Defaults' not in config:
            config['Model Defaults'] = {}
        model_section = config['Model Defaults']
        model_section_keys = {k.upper(): k for k in model_section.keys()}
        
        # Load model values from Model Defaults
        for key, default_value in instance.get_original_defaults().items():
            if key.startswith('DEFAULT_') and any(model_type in key.lower() for model_type in ['transformer', 'text_encoder', 'tokenizer', 'vae', 'feature_extractor', 'image_encoder']):
                try:
                    # Look up the actual key in a case-insensitive way
                    actual_key = model_section_keys.get(key.upper(), key)
                    if actual_key not in model_section:
                        actual_key = model_section_keys.get(key.lower(), key)
                    
                    value = model_section.get(actual_key, str(default_value))
                    # Remove quotes if present
                    parsed_value = str(value).strip("'").strip('"')
                    if parsed_value.lower() == 'none':
                        parsed_value = default_value
                    setattr(instance, key, parsed_value)
                except Exception as e:
                    debug_print(f"Error loading model setting {key}: {str(e)}, using default value: {default_value}")
                    setattr(instance, key, default_value)
                    model_section[key] = str(default_value)
                    save_settings(config)

        # Load System Defaults section
        if 'System Defaults' not in config:
            config['System Defaults'] = {}
        section = config['System Defaults']
        section_keys = {k.upper(): k for k in section.keys()}
        
        # Load system values from settings, using defaults as fallback
        for key, default_value in instance.get_original_defaults().items():
            if not key.startswith('DEFAULT_'):
                try:
                    # Look up the actual key in a case-insensitive way
                    actual_key = section_keys.get(key.upper(), key)
                    if actual_key not in section:
                        actual_key = section_keys.get(key.lower(), key)
                    
                    # Special handling for HF_TOKEN - only use default if not present
                    if key == 'HF_TOKEN':
                        if 'hf_token' not in section and 'HF_TOKEN' not in section:
                            value = str(default_value)
                        else:
                            value = section.get(actual_key, section.get('hf_token', section.get('HF_TOKEN')))
                    else:
                        value = section.get(actual_key, str(default_value))
                    
                    # Handle different types appropriately
                    if isinstance(default_value, bool):
                        parsed_value = str(value).lower() in ('true', 't', 'yes', 'y', '1')
                    elif isinstance(default_value, (int, float)):
                        parsed_value = type(default_value)(str(value).strip("'"))
                    else:
                        parsed_value = str(value).strip("'")
                    
                    setattr(instance, key, parsed_value)
                except Exception as e:
                    debug_print(f"Error loading system setting {key}: {str(e)}, using default value: {default_value}")
                    # Only set default for HF_TOKEN if it doesn't exist
                    if key == 'HF_TOKEN' and ('hf_token' in section or 'HF_TOKEN' in section):
                        continue
                    setattr(instance, key, default_value)
                    section[key] = str(default_value)
                    save_settings(config)

        return instance

    @classmethod
    def to_settings(cls, config):
        """Save Config instance values to settings.ini"""
        # Save Job Defaults section
        section = config['Job Defaults']
        job_defaults = [
            'DEFAULT_PROMPT', 'DEFAULT_N_PROMPT', 'DEFAULT_VIDEO_LENGTH',
            'DEFAULT_GS', 'DEFAULT_STEPS', 'DEFAULT_USE_TEACACHE', 'DEFAULT_SEED',
            'DEFAULT_CFG', 'DEFAULT_RS', 'DEFAULT_GPU_MEMORY', 'DEFAULT_MP4_CRF',
            'DEFAULT_KEEP_TEMP_PNG', 'DEFAULT_KEEP_TEMP_JSON'
        ]
        for key in job_defaults:
            section[key] = repr(getattr(cls, key))

        # Save Model Defaults section
        if 'Model Defaults' not in config:
            config['Model Defaults'] = {}
        section = config['Model Defaults']
        model_defaults = [
            'DEFAULT_TRANSFORMER', 'DEFAULT_TEXT_ENCODER', 'DEFAULT_TEXT_ENCODER_2',
            'DEFAULT_TOKENIZER', 'DEFAULT_TOKENIZER_2', 'DEFAULT_VAE',
            'DEFAULT_FEATURE_EXTRACTOR', 'DEFAULT_IMAGE_ENCODER'
        ]
        for key in model_defaults:
            section[key] = repr(getattr(cls, key))

        # Save System Defaults section
        if 'System Defaults' not in config:
            config['System Defaults'] = {}
        section = config['System Defaults']
        system_defaults = ['OUTPUTS_FOLDER', 'JOB_HISTORY_FOLDER', 'DEBUG_MODE', 'KEEP_TEMP_MP4', 'KEEP_COMPLETED_JOB']
        for key in system_defaults:
            section[key] = str(getattr(cls, key))

        save_settings(config)

    @classmethod
    def get_default_prompt_tuple(cls):
        """Returns a tuple of all default values in the correct order"""
        return (
            cls.DEFAULT_PROMPT,
            cls.DEFAULT_N_PROMPT,
            cls.DEFAULT_JOB_NAME,
            cls.DEFAULT_VIDEO_LENGTH,
            cls.DEFAULT_GS,
            cls.DEFAULT_STEPS,
            cls.DEFAULT_USE_TEACACHE,
            cls.DEFAULT_SEED,
            cls.DEFAULT_CFG,
            cls.DEFAULT_RS,
            cls.DEFAULT_GPU_MEMORY,
            cls.DEFAULT_MP4_CRF,
            cls.DEFAULT_KEEP_TEMP_PNG,
            cls.DEFAULT_KEEP_TEMP_JSON
        )

    @classmethod
    def get_default_prompt_dict(cls):
        """Returns a dictionary of default values for quick prompts"""
        return {
            'prompt': cls.DEFAULT_PROMPT,
            'n_prompt': cls.DEFAULT_N_PROMPT,
            'job_name': cls.DEFAULT_JOB_NAME,
            'length': cls.DEFAULT_VIDEO_LENGTH,
            'gs': cls.DEFAULT_GS,
            'steps': cls.DEFAULT_STEPS,
            'use_teacache': cls.DEFAULT_USE_TEACACHE,
            'seed': cls.DEFAULT_SEED,
            'cfg': cls.DEFAULT_CFG,
            'rs': cls.DEFAULT_RS,
            'gpu_memory': cls.DEFAULT_GPU_MEMORY,
            'mp4_crf': cls.DEFAULT_MP4_CRF,
            'keep_temp_png': cls.DEFAULT_KEEP_TEMP_PNG,
            'keep_temp_json': cls.DEFAULT_KEEP_TEMP_JSON
        }

def load_settings():
    """Load settings from settings.ini file and ensure all sections and values exist"""
    config = configparser.ConfigParser()
    
    # Get default values
    default_values = Config.get_original_defaults()
    
    # Create default sections if file doesn't exist
    if not os.path.exists(INI_FILE):
        # Split defaults into appropriate sections
        job_defaults = {k: v for k, v in default_values.items() 
                       if k.startswith('DEFAULT_') and not any(model_type in k.lower() 
                       for model_type in ['transformer', 'text_encoder', 'tokenizer', 'vae', 'feature_extractor', 'image_encoder'])}
        
        model_defaults = {k: v for k, v in default_values.items() 
                         if k.startswith('DEFAULT_') and any(model_type in k.lower() 
                         for model_type in ['transformer', 'text_encoder', 'tokenizer', 'vae', 'feature_extractor', 'image_encoder'])}
        
        system_defaults = {k: v for k, v in default_values.items() if not k.startswith('DEFAULT_')}
        
        config['Job Defaults'] = {k: repr(v) for k, v in job_defaults.items()}
        config['Model Defaults'] = {k: repr(v) for k, v in model_defaults.items()}
        config['System Defaults'] = {k: str(v) for k, v in system_defaults.items()}
        
        with open(INI_FILE, 'w') as f:
            config.write(f)
    else:
        # Read existing config
        config.read(INI_FILE)
        
        # Ensure Job Defaults section exists with all non-model values
        if 'Job Defaults' not in config:
            config['Job Defaults'] = {}
        
        # Check and add any missing non-model values in Job Defaults
        for key, value in default_values.items():
            if (key.startswith('DEFAULT_') and 
                not any(model_type in key.lower() for model_type in 
                    ['transformer', 'text_encoder', 'tokenizer', 'vae', 'feature_extractor', 'image_encoder'])):
                if key not in config['Job Defaults'] or config['Job Defaults'][key].strip("'").strip('"').lower() == 'none':
                    config['Job Defaults'][key] = repr(value)
        
        # Ensure Model Defaults section exists with all model values
        if 'Model Defaults' not in config:
            config['Model Defaults'] = {}
        
        # Check and add any missing model values or replace None values
        for key, value in default_values.items():
            if (key.startswith('DEFAULT_') and 
                any(model_type in key.lower() for model_type in 
                    ['transformer', 'text_encoder', 'tokenizer', 'vae', 'feature_extractor', 'image_encoder'])):
                if key not in config['Model Defaults'] or config['Model Defaults'][key].strip("'").strip('"').lower() == 'none':
                    config['Model Defaults'][key] = repr(value)
                    debug_print(f"Replacing None value for {key} with default: {value}")
        
        # Ensure System Defaults section exists with all values
        if 'System Defaults' not in config:
            config['System Defaults'] = {}
        
        # Check and add any missing system values
        for key, value in default_values.items():
            if not key.startswith('DEFAULT_'):
                if key not in config['System Defaults'] or config['System Defaults'][key].strip("'").strip('"').lower() == 'none':
                    config['System Defaults'][key] = str(value)
        
        # Save any changes made to the config
        with open(INI_FILE, 'w') as f:
            config.write(f)
    
    return config

def save_settings_from_ui(outputs_folder, job_history_folder, debug_mode, keep_temp_mp4, keep_completed_job,
                         transformer, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                         vae, feature_extractor, image_encoder):
    """Save settings from UI inputs"""
    settings_config = configparser.ConfigParser()
    
    # Create required sections
    settings_config['Job Defaults'] = {}
    settings_config['Model Defaults'] = {}
    settings_config['System Defaults'] = {}
    
    # System Defaults
    settings_config['System Defaults'] = {
        'OUTPUTS_FOLDER': repr(outputs_folder),
        'JOB_HISTORY_FOLDER': repr(job_history_folder),
        'DEBUG_MODE': repr(debug_mode),
        'KEEP_TEMP_MP4': repr(keep_temp_mp4),
        'KEEP_COMPLETED_JOB': repr(keep_completed_job)
    }
    
    # Model Defaults
    settings_config['Model Defaults'] = {
        'DEFAULT_TRANSFORMER': repr(transformer),
        'DEFAULT_TEXT_ENCODER': repr(text_encoder),
        'DEFAULT_TEXT_ENCODER_2': repr(text_encoder_2),
        'DEFAULT_TOKENIZER': repr(tokenizer),
        'DEFAULT_TOKENIZER_2': repr(tokenizer_2),
        'DEFAULT_VAE': repr(vae),
        'DEFAULT_FEATURE_EXTRACTOR': repr(feature_extractor),
        'DEFAULT_IMAGE_ENCODER': repr(image_encoder)
    }
    
    # Update global Config object
    Config.OUTPUTS_FOLDER = outputs_folder
    Config.JOB_HISTORY_FOLDER = job_history_folder
    Config.DEBUG_MODE = debug_mode
    Config.KEEP_COMPLETED_JOB = keep_completed_job
    Config.DEFAULT_TRANSFORMER = transformer
    Config.DEFAULT_TEXT_ENCODER = text_encoder
    Config.DEFAULT_TEXT_ENCODER_2 = text_encoder_2
    Config.DEFAULT_TOKENIZER = tokenizer
    Config.DEFAULT_TOKENIZER_2 = tokenizer_2
    Config.DEFAULT_VAE = vae
    Config.DEFAULT_FEATURE_EXTRACTOR = feature_extractor
    Config.DEFAULT_IMAGE_ENCODER = image_encoder
    
    # Load existing settings to preserve other values
    existing_config = load_settings()
    
    # Update with new settings
    for section in settings_config.sections():
        if section not in existing_config:
            existing_config[section] = {}
        for key, value in settings_config[section].items():
            existing_config[section][key] = value
    
    # Save updated settings
    save_settings(existing_config)
    return "Settings saved successfully. Restart required for changes to take effect."


def restore_job_defaults():
    """Restore Job Defaults to original values"""
    # Get the original defaults
    defaults = Config.get_original_defaults()
    debug_print("Restoring original defaults:")
    
    # Load the config file
    config = load_settings()
    if 'Job Defaults' not in config:
        config['Job Defaults'] = {}
    section = config['Job Defaults']
    
    # Update the section with all non-model default values
    for key, value in defaults.items():
        if (key.startswith('DEFAULT_') and 
            not any(model_type in key.lower() for model_type in 
                ['transformer', 'text_encoder', 'tokenizer', 'vae', 'feature_extractor', 'image_encoder'])):
            debug_print(f"  Setting {key} = {value}")
            section[key] = str(value)
    
    # Save to settings.ini
    save_settings(config)
    debug_print("Saved original defaults to settings.ini")
    
    # Return values in the order expected by the UI
    return [
        defaults['DEFAULT_PROMPT'],
        defaults['DEFAULT_N_PROMPT'],
        defaults['DEFAULT_USE_TEACACHE'],
        defaults['DEFAULT_SEED'],
        defaults['DEFAULT_JOB_NAME'],
        defaults['DEFAULT_VIDEO_LENGTH'],
        defaults['DEFAULT_STEPS'],
        defaults['DEFAULT_CFG'],
        defaults['DEFAULT_GS'],
        defaults['DEFAULT_RS'],
        defaults['DEFAULT_GPU_MEMORY'],
        defaults['DEFAULT_MP4_CRF'],
        defaults['DEFAULT_KEEP_TEMP_PNG'],
        defaults['DEFAULT_KEEP_TEMP_JSON']
    ]

def save_queue():
    try:
        jobs = [job.to_dict() for job in job_queue]
        with open(QUEUE_JSON_FILE, 'w') as f:
            json.dump(jobs, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving queue: {str(e)}")
        traceback.print_exc()
        return False

def load_queue():
    try:
        if os.path.exists(QUEUE_JSON_FILE):
            try:
                with open(QUEUE_JSON_FILE, 'r') as f:
                    jobs = json.load(f)
            except json.JSONDecodeError as e:
                alert_print(f"Error reading queue file (corrupted JSON): {str(e)}")
                # Try to load a backup or create empty queue
                backup_file = QUEUE_JSON_FILE + '.backup'
                if os.path.exists(backup_file):
                    debug_print("Attempting to load from backup file...")
                    try:
                        with open(backup_file, 'r') as f:
                            jobs = json.load(f)
                    except:
                        debug_print("Backup file also corrupted, starting with empty queue")
                        jobs = []
                else:
                    debug_print("No backup file found, starting with empty queue")
                    jobs = []
            
            # Clear existing queue and load valid jobs from file
            job_queue.clear()
            valid_jobs = []
            for job_data in jobs:
                job = QueuedJob.from_dict(job_data)
                if job is not None:
                    valid_jobs.append(job)
                else:
                    debug_print(f"Skipping invalid job data: {job_data}")
            
            # Only update queue if we have valid jobs
            if valid_jobs:
                job_queue.extend(valid_jobs)
                debug_print(f"Loaded {len(valid_jobs)} valid jobs")
            
            return job_queue
        else:
            debug_print("No queue file found")
            return []
    except Exception as e:
        alert_print(f"Error loading queue: {str(e)}")
        debug_print(f"Error details: {traceback.format_exc()}")
        return []

def setup_local_variables():
    """Set up local variables from Config values"""
    global job_history_folder, outputs_folder, debug_mode, keep_temp_mp4, keep_completed_job
    job_history_folder = Config.JOB_HISTORY_FOLDER
    outputs_folder = Config.OUTPUTS_FOLDER
    debug_mode = Config.DEBUG_MODE
    keep_temp_mp4 = Config.KEEP_TEMP_MP4
    keep_completed_job = Config.KEEP_COMPLETED_JOB

# Initialize settings first
settings_config = load_settings()
Config = Config.from_settings(settings_config)
print("Loaded settings into Config:")


# Create necessary directories using values from Config
os.makedirs(Config.OUTPUTS_FOLDER, exist_ok=True)
os.makedirs(Config.JOB_HISTORY_FOLDER, exist_ok=True)


# Initialize job queue as a list
job_queue = []

# Set up local variables
setup_local_variables()

# Initialize quick prompts
DEFAULT_PROMPTS = [
    Config.get_default_prompt_dict(),
    {
        'prompt': 'A character doing some simple body movements.',
        'n_prompt': '',
        'job_name': Config.DEFAULT_JOB_NAME,
        'length': Config.DEFAULT_VIDEO_LENGTH,
        'gs': Config.DEFAULT_GS,
        'steps': Config.DEFAULT_STEPS,
        'use_teacache': Config.DEFAULT_USE_TEACACHE,
        'seed': Config.DEFAULT_SEED,
        'cfg': Config.DEFAULT_CFG,
        'rs': Config.DEFAULT_RS,
        'gpu_memory': Config.DEFAULT_GPU_MEMORY,
        'mp4_crf': Config.DEFAULT_MP4_CRF,
        'keep_temp_png': Config.DEFAULT_KEEP_TEMP_PNG,
        'keep_temp_json': Config.DEFAULT_KEEP_TEMP_JSON
    }
]

# Load existing prompts or create the file with defaults
if os.path.exists(QUICK_LIST_FILE):
    with open(QUICK_LIST_FILE, 'r') as f:
        quick_prompts = json.load(f)
else:
    quick_prompts = DEFAULT_PROMPTS.copy()
    with open(QUICK_LIST_FILE, 'w') as f:
        json.dump(quick_prompts, f, indent=2)

@dataclass
class QueuedJob:
    prompt: str
    image_path: str
    job_name: str
    video_length: float
    seed: int
    use_teacache: bool
    gpu_memory: float
    steps: int
    cfg: float
    gs: float
    rs: float
    n_prompt: str
    status: str = "pending"
    thumbnail: str = ""
    mp4_crf: float = 16
    keep_temp_png: bool = False
    keep_temp_json: bool = False
    outputs_folder: str = str(Config.OUTPUTS_FOLDER)  # Keep original name
    job_history_folder: str = str(Config.JOB_HISTORY_FOLDER)  # Keep original name
    keep_temp_mp4: bool = Config.KEEP_TEMP_MP4
    keep_completed_job: bool = Config.KEEP_COMPLETED_JOB

    def to_dict(self):
        try:
            return {
                'prompt': self.prompt,
                'image_path': self.image_path,
                'job_name': self.job_name,
                'video_length': self.video_length,
                'seed': self.seed,
                'use_teacache': self.use_teacache,
                'gpu_memory': self.gpu_memory,
                'steps': self.steps,
                'cfg': self.cfg,
                'gs': self.gs,
                'rs': self.rs,
                'n_prompt': self.n_prompt,
                'status': self.status,
                'thumbnail': self.thumbnail,
                'mp4_crf': self.mp4_crf,
                'keep_temp_png': self.keep_temp_png,
                'keep_temp_json': self.keep_temp_json,
                'outputs_folder': str(self.outputs_folder),
                'job_history_folder': str(self.job_history_folder),
                'keep_temp_mp4': self.keep_temp_mp4,
                'keep_completed_job': self.keep_completed_job
            }
        except Exception as e:
            alert_print(f"Error converting job to dict: {str(e)}")
            return None

    @classmethod
    def from_dict(cls, data):
        try:
            return cls(
                prompt=data.get('prompt', ''),
                image_path=data.get('image_path', ''),
                job_name=data.get('job_name', ''),
                video_length=float(data.get('video_length', 5.0)),
                seed=int(data.get('seed', -1)),
                use_teacache=bool(data.get('use_teacache', True)),
                gpu_memory=float(data.get('gpu_memory', 6.0)),
                steps=int(data.get('steps', 25)),
                cfg=float(data.get('cfg', 1.0)),
                gs=float(data.get('gs', 10.0)),
                rs=float(data.get('rs', 0.0)),
                n_prompt=data.get('n_prompt', ''),
                status=data.get('status', 'pending'),
                thumbnail=data.get('thumbnail', ''),
                mp4_crf=float(data.get('mp4_crf', 16)),
                keep_temp_png=bool(data.get('keep_temp_png', False)),
                keep_temp_json=bool(data.get('keep_temp_json', False)),
                outputs_folder=str(data.get('outputs_folder', Config.OUTPUTS_FOLDER)),
                job_history_folder=str(data.get('job_history_folder', Config.JOB_HISTORY_FOLDER)),
                keep_temp_mp4=bool(data.get('keep_temp_mp4', Config.KEEP_TEMP_MP4)),
                keep_completed_job=bool(data.get('keep_completed_job', Config.KEEP_COMPLETED_JOB))
            )
        except Exception as e:
            alert_print(f"Error creating job from dict: {str(e)}")
            debug_print(f"Problem data: {data}")
            debug_print(f"Error details: {traceback.format_exc()}")
            return None

def save_image_to_temp(image: np.ndarray, job_name: str) -> str:
    """Save image to temp directory and return the path"""
    try:
        # Handle Gallery tuple format
        if isinstance(image, tuple):
            image = image[0]  # Get the file path from the tuple
        if isinstance(image, str):
            # If it's a path, open the image
            pil_image = Image.open(image)
            # Only convert if it's RGBA
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
        else:
            # If it's already a numpy array
            pil_image = Image.fromarray(image)
            # Only convert if it's RGBA
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
            
        filename = f"queue_image_{job_name}.png"
        filepath = os.path.join(temp_queue_images, filename)
        # Save image
        pil_image.save(filepath)
        return filepath
    except Exception as e:
        alert_print(f"Error saving image: {str(e)}")
        traceback.print_exc()
        return ""

def add_to_queue(prompt, n_prompt, input_image, video_length, seed, job_name, use_teacache, gpu_memory, steps, cfg, gs, rs, mp4_crf, keep_temp_png, keep_temp_json, create_job_outputs_folder=None, create_job_history_folder=None, keep_temp_mp4=None, create_job_keep_completed_job=None, status="pending"):
    """Add a new job to the queue"""
    try:
        # Set default values for optional parameters
        if create_job_outputs_folder is None:
            create_job_outputs_folder = Config.OUTPUTS_FOLDER
        if create_job_history_folder is None:
            create_job_history_folder = Config.JOB_HISTORY_FOLDER
        if keep_temp_mp4 is None:
            keep_temp_mp4 = Config.KEEP_TEMP_MP4
        if create_job_keep_completed_job is None:
            create_job_keep_completed_job = Config.KEEP_COMPLETED_JOB
               
        hex_id = uuid.uuid4().hex[:8]
        job_name = f"{job_name}_{hex_id}"
        load_queue()

        # Handle text-to-video case
        if input_image is None:
            job = QueuedJob(
                prompt=prompt,
                image_path="text2video",  # Set to None for text-to-video
                job_name=job_name,
                video_length=video_length,
                seed=seed,
                use_teacache=use_teacache,
                gpu_memory=gpu_memory,
                steps=steps,
                cfg=cfg,
                gs=gs,
                rs=rs,
                n_prompt=n_prompt,
                status=status,
                mp4_crf=mp4_crf,
                keep_temp_png=keep_temp_png,
                keep_temp_json=keep_temp_json,
                outputs_folder=create_job_outputs_folder,  # Map to job's outputs_folder
                job_history_folder=create_job_history_folder,  # Map to job's job_history_folder
                keep_temp_mp4=keep_temp_mp4,
                keep_completed_job=create_job_keep_completed_job  # Map to job's keep_completed_job
            )
            # Find the first completed job
            insert_index = len(job_queue)
            for i, existing_job in enumerate(job_queue):
                if existing_job.status == "completed":
                    insert_index = i
                    break
            job_queue.insert(insert_index, job)
            job.thumbnail = create_thumbnail(job, status_change=True)  
            save_queue()
            debug_print(f"Total jobs in the queue:{len(job_queue)}")
            return job_name

        # Handle image-to-video case
        if isinstance(input_image, np.ndarray):
            # Save the input image
            image_path = save_image_to_temp(input_image, job_name)
            if image_path == "text2video":
                return None

            job = QueuedJob(
                prompt=prompt,
                image_path=image_path,
                video_length=video_length,
                job_name=job_name,
                seed=seed,
                use_teacache=use_teacache,
                gpu_memory=gpu_memory,
                steps=steps,
                cfg=cfg,
                gs=gs,
                rs=rs,
                n_prompt=n_prompt,
                status=status,
                thumbnail="",
                mp4_crf=mp4_crf,
                keep_temp_png=keep_temp_png,
                keep_temp_json=keep_temp_json,
                outputs_folder=create_job_outputs_folder,  # Map to job's outputs_folder
                job_history_folder=create_job_history_folder,  # Map to job's job_history_folder
                keep_temp_mp4=keep_temp_mp4,
                keep_completed_job=create_job_keep_completed_job  # Map to job's keep_completed_job
            )
            # Find the first completed job
            insert_index = len(job_queue)
            for i, existing_job in enumerate(job_queue):
                if existing_job.status == "completed":
                    insert_index = i
                    break
            job_queue.insert(insert_index, job)
            job.thumbnail = create_thumbnail(job, status_change=False)
            save_queue()
            debug_print(f"Total jobs in the queue:{len(job_queue)}")
            return job_name
        else:
            alert_print("Invalid input image format")
            return None
    except Exception as e:
        alert_print(f"Error adding to queue: {str(e)}")
        return None

def create_thumbnail(job, status_change=False):
    """Create a thumbnail for a job"""
    # Add status
    status_color = {
        "pending": "yellow",
        "processing": "blue",
        "completed": "green",
        "failed": "red"
    }.get(job.status, "white")
    
    # Initialize status overlay
    status_overlay = "RUNNING" if job.status == "processing" else ("DONE" if job.status == "completed" else job.status.upper())
    
    try:
        # Try to load arial font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            small_font = ImageFont.truetype("arial.ttf", 12)
        except (OSError, IOError):
            try:
                # DejaVuSans ships with Pillow and is usually available
                font = ImageFont.truetype("DejaVuSans.ttf", 16)
                small_font = ImageFont.truetype("DejaVuSans.ttf", 12)
            except (OSError, IOError):
                # Final fallback to a simple built-in bitmap font
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()

        # Handle text-to-video case (job.image_path is text2video)
        if job.image_path == "text2video":
            debug_print(f"in create_thumbnail {job.job_name} has a job.image_path that is {job.image_path}")
            if not job.thumbnail or status_change:  # Create new thumbnail if none exists or status changed
                # Create a text-to-video thumbnail
                img = Image.new('RGB', (200, 200), color='black')
                draw = ImageDraw.Draw(img)
                # Calculate text positions using the same approach as the example
                text1 = "Text to Video"
                text2 = "Generation"
                text3 = status_overlay
                
                # Get text sizes
                text1_bbox = draw.textbbox((0, 0), text1, font=font)
                text2_bbox = draw.textbbox((0, 0), text2, font=font)
                text3_bbox = draw.textbbox((0, 0), text3, font=font)
                
                # Calculate positions to center text
                x1 = (200 - (text1_bbox[2] - text1_bbox[0])) // 2
                x2 = (200 - (text2_bbox[2] - text2_bbox[0])) // 2
                x3 = (200 - (text3_bbox[2] - text3_bbox[0])) // 2
                
                # Add text-to-video indicator with calculated positions
                draw.text((x1, 80), text1, fill='white', font=font)
                draw.text((x2, 100), text2, fill='white', font=font)
                draw.text((x3, 120), text3, fill=status_color, font=font)
                
                # Save thumbnail
                thumbnail_path = os.path.join(temp_queue_images, f"thumb_{job.job_name}.png")
                img.save(thumbnail_path)
                debug_print(f"thumbnail saved {thumbnail_path}")
                job.thumbnail = thumbnail_path
                save_queue()
            return job.thumbnail

        # Handle missing image-based cases 
        if job.image_path != "text2video" and not os.path.exists(job.image_path) and not job.thumbnail:
            # Create missing image thumbnail
            img = Image.new('RGB', (200, 200), color='black')
            draw = ImageDraw.Draw(img)
            
            # Calculate text position for "MISSING IMAGE"
            text = "MISSING IMAGE"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            x = (200 - (text_bbox[2] - text_bbox[0])) // 2
            y = (200 - (text_bbox[3] - text_bbox[1])) // 2
            
            # Add black outline to make text more readable
            for offset in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                draw.text((x+offset[0], y+offset[1]), text, font=font, fill=(0,0,0))
            draw.text((x, y), text, fill='red', font=font)
            
            # Save thumbnail
            thumbnail_path = os.path.join(temp_queue_images, f"thumb_{job.job_name}.png")
            img.save(thumbnail_path)
            job.thumbnail = thumbnail_path
            save_queue()
            return thumbnail_path

        # Normal case - create thumbnail from existing image add overlay
        if job.image_path != "text2video":
            img = Image.open(job.image_path)
            width, height = img.size
            new_height = 200
            new_width = int((new_height / height) * width)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # Create a new image with padding
            new_img = Image.new('RGB', (200, 200), color='black')
            new_img.paste(img, ((200 - img.width) // 2, (200 - img.height) // 2))
            # Add status text if provided
            if status_change:
                draw = ImageDraw.Draw(new_img)
                # Calculate text position for status overlay
                text_bbox = draw.textbbox((0, 0), status_overlay, font=font)
                x = (200 - (text_bbox[2] - text_bbox[0])) // 2
                y = (200 - (text_bbox[3] - text_bbox[1])) // 2
                
                # Add black outline to make text more readable
                for offset in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                    draw.text((x+offset[0], y+offset[1]), status_overlay, font=font, fill=(0,0,0))
                draw.text((x, y), status_overlay, fill=status_color, font=font)
            
            # Save thumbnail
            thumbnail_path = os.path.join(temp_queue_images, f"thumb_{job.job_name}.png")
            new_img.save(thumbnail_path)
            debug_print(f"thumbnail saved {thumbnail_path}")
            job.thumbnail = thumbnail_path
            save_queue()
        return thumbnail_path
    except Exception as e:
        alert_print(f"Error creating thumbnail: {str(e)}")
        return ""

def update_queue_display():
    """Update the queue display with current jobs from JSON"""
    try:
        queue_data = []
        for job in job_queue:
            # Only check for missing images if the job is not being deleted
            if job.status != "deleting":
                # Check if both queue image and thumbnail are missing
                queue_image_missing = not os.path.exists(job.image_path) if job.image_path else True
                thumbnail_missing = not os.path.exists(job.thumbnail) if job.thumbnail else True
                #######
                if queue_image_missing and thumbnail_missing:
                    # Create missing placeholder images
                    new_thumbnail = create_thumbnail(job, status_change=False)
                elif not job.thumbnail and job.image_path:
                    job.thumbnail = create_thumbnail(job, status_change=False)

            # Add job data to display
            if job.thumbnail:
                caption = f"{job.prompt} \n Negative: {job.n_prompt}\nLength: {job.video_length}s\nGS: {job.gs}"
                queue_data.append((job.thumbnail, caption))
        
        return queue_data
    except Exception as e:
        alert_print(f"Error updating queue display: {str(e)}")
        traceback.print_exc()
        return []

def update_queue_table():
    """Update the queue table display with current jobs from JSON"""
    data = []
    for job in job_queue:
        # Add job data to display
        if not job.thumbnail or not os.path.exists(job.thumbnail):
            create_thumbnail(job, status_change=True)
            
        try:
            # Read the image and convert to base64
            with open(job.thumbnail, "rb") as img_file:
                import base64
                img_data = base64.b64encode(img_file.read()).decode()
            img_md = f'<div style="text-align: center; font-size: 0.8em; color: #666; margin-bottom: 5px;">{job.status}</div><div style="text-align: center; font-size: 0.8em; color: #666;">{job.job_name}</div><div style="text-align: center; font-size: 0.8em; color: #666;">seed: {job.seed}</div><div style="text-align: center; font-size: 0.8em; color: #666;">Length: {job.video_length:.1f}s</div><img src="data:image/png;base64,{img_data}" alt="Input" style="max-width:100px; max-height:100px; display: block; margin: auto; object-fit: contain; transform: scale(0.75); transform-origin: top left;" />'
        except Exception as e:
            alert_print(f"Error converting image to base64: {str(e)}")
            img_md = ""

        # Use full prompt text without truncation
        prompt_cell = f'<span style="white-space: normal; word-wrap: break-word; display: block; width: 100%;">{job.prompt}</span>'

        # Add edit button for pending jobs
        edit_button = "✎" if job.status in ["pending", "completed"] else ""
        top_button = "⏫️"
        up_button = "⬆️"
        down_button = "⬇️"
        bottom_button = "⏬️"
        remove_button = "❌"
        copy_button = "📋"

        data.append([
            img_md,           # Input thumbnail with ID, length, and status
            top_button,
            up_button,
            down_button,
            bottom_button,
            remove_button,
            copy_button,
            edit_button,
            prompt_cell      # Prompt
        ])
    return gr.DataFrame(value=data, visible=True, elem_classes=["gradio-dataframe"])

def cleanup_orphaned_files():
    print ("Clean up any temp files that don't correspond to jobs in the queue")
    try:
        # Get all job files from queue
        job_files = set()
        for job in job_queue:
            if job.image_path:
                job_files.add(job.image_path)
            if job.thumbnail:
                job_files.add(job.thumbnail)
        
        # Get all files in temp directory
        temp_files = set()
        for root, _, files in os.walk(temp_queue_images):
            for file in files:
                temp_files.add(os.path.join(root, file))
        
        # Find orphaned files (in temp but not in queue)
        orphaned_files = temp_files - job_files
        
        # Delete orphaned files
        for file in orphaned_files:
            try:
                os.remove(file)
            except Exception as e:
                alert_print(f"Error deleting file {file}: {str(e)}")
    except Exception as e:
        alert_print(f"Error in cleanup_orphaned_files: {str(e)}")
        traceback.print_exc()


def reset_processing_jobs():
    print ("Reset any processing to pending and move them to top of queue")
    global job_queue
    try:
        # First load the queue from JSON
        load_queue()
        
        # Remove completed jobs if keep_completed_job is False
        if not keep_completed_job:
            completed_jobs_to_remove = [
                job for job in job_queue 
                if job.status == "completed" 
                and (
                    (hasattr(job, 'keep_completed_job') and not job.keep_completed_job) or
                    (not hasattr(job, 'keep_completed_job') and not keep_completed_job)
                )
            ]

            # Count jobs that will be removed
            completed_jobs_count = len(completed_jobs_to_remove)

            # Remove the jobs that meet our criteria
            if completed_jobs_count > 0:
                job_queue = [job for job in job_queue if job not in completed_jobs_to_remove]
                debug_print(f"Removed {completed_jobs_count} completed jobs from queue")
        
        # Find all processing jobs and move them to top
        processing_jobs = []
        for job in job_queue:
            if job.status == "processing":
                debug_print(f"Found job {job.job_name} with status {job.status}")
                mark_job_pending(job)
                processing_jobs.append(job)
        
        # Remove these jobs from their current positions
        for job in processing_jobs:
            if job in job_queue:
                job_queue.remove(job)
        
        # Add them back at the top in reverse order (so they maintain their relative order)
        for job in reversed(processing_jobs):
            job_queue.insert(0, job)
            debug_print(f"marked previously aborted job as pending and Moved job {job.job_name} to top of queue")
        
        # Find all failed jobs and move them to top
        failed_jobs = []
        for job in job_queue:
            if job.status == "failed":
                debug_print(f"Found job {job.job_name} with status {job.status}")
                mark_job_pending(job)
                failed_jobs.append(job)
        
        # Remove these jobs from their current positions
        for job in failed_jobs:
            if job in job_queue:
                job_queue.remove(job)
        
        # Add them back at the top in reverse order (so they maintain their relative order)
        for job in reversed(failed_jobs):
            job_queue.insert(0, job)
            debug_print(f"marked previously aborted job as pending and Moved job {job.job_name} to top of queue")
        
        save_queue()
        debug_print(f"{len(processing_jobs)} aborted jobs found and moved to the top as pending")
        debug_print(f"{len(failed_jobs)} failed jobs found and moved to the top as pending")
        debug_print(f"Total jobs in the queue:{len(job_queue)}")
    except Exception as e:
        alert_print(f"Error resetting processing jobs: {str(e)}")

# Quick prompts management functions
def get_default_prompt():
    try:
        if quick_prompts and len(quick_prompts) > 0:
            return (
                quick_prompts[0]['prompt'],
                quick_prompts[0]['n_prompt'],
                quick_prompts[0].get('job_name', Config.DEFAULT_JOB_NAME),
                quick_prompts[0]['video_length'],
                quick_prompts[0].get('gs', Config.DEFAULT_GS),
                quick_prompts[0].get('steps', Config.DEFAULT_STEPS),
                quick_prompts[0].get('use_teacache', Config.DEFAULT_USE_TEACACHE),
                quick_prompts[0].get('seed', Config.DEFAULT_SEED),
                quick_prompts[0].get('cfg', Config.DEFAULT_CFG),
                quick_prompts[0].get('rs', Config.DEFAULT_RS),
                quick_prompts[0].get('gpu_memory', Config.DEFAULT_GPU_MEMORY),
                quick_prompts[0].get('mp4_crf', Config.DEFAULT_MP4_CRF),
                quick_prompts[0].get('keep_temp_png', Config.DEFAULT_KEEP_TEMP_PNG),
                quick_prompts[0].get('keep_temp_json', Config.DEFAULT_KEEP_TEMP_JSON)
            )
        return Config.get_default_prompt_tuple()
    except Exception as e:
        alert_print(f"Error getting default prompt: {str(e)}")
        return Config.get_default_prompt_tuple()

def save_quick_prompt(prompt_text, n_prompt_text_value, video_length_value, job_name_value, gs_value, steps_value, use_teacache_value, seed_value, cfg_value, rs_value, gpu_memory_value, mp4_crf_value, keep_temp_png_value, keep_temp_json_value):
    global quick_prompts
    if prompt_text:
        # Check if prompt already exists
        for item in quick_prompts:
            if item['prompt'] == prompt_text:
                item['n_prompt'] = n_prompt_text_value
                item['video_length'] = video_length_value
                item['job_name'] = job_name_value
                item['gs'] = gs_value
                item['steps'] = steps_value
                item['use_teacache'] = use_teacache_value
                item['seed'] = seed_value
                item['cfg'] = cfg_value
                item['rs'] = rs_value
                item['gpu_memory'] = gpu_memory_value
                item['mp4_crf'] = mp4_crf_value
                item['keep_temp_png'] = keep_temp_png_value
                item['keep_temp_json'] = keep_temp_json_value
                break
        else:
            quick_prompts.append({
                'prompt': prompt_text,
                'n_prompt': n_prompt_text_value,
                'video_length': video_length_value,
                'job_name': job_name_value,
                'gs': gs_value,
                'steps': steps_value,
                'use_teacache': use_teacache_value,
                'seed': seed_value,
                'cfg': cfg_value,
                'rs': rs_value,
                'gpu_memory': gpu_memory_value,
                'mp4_crf': mp4_crf_value,
                'keep_temp_png': keep_temp_png_value,
                'keep_temp_json': keep_temp_json_value
            })
        
        with open(QUICK_LIST_FILE, 'w') as f:
            json.dump(quick_prompts, f, indent=2)

        # Return all values with gr.update() for all components
        return (
            prompt_text,  # prompt (no update needed, just return value)
            gr.update(choices=[item['prompt'] for item in quick_prompts], value=prompt_text),  # quick_list
            n_prompt_text_value,  # n_prompt
            video_length_value,  # video_length
            job_name_value,  # job_name
            gs_value,  # gs
            steps_value,  # steps
            use_teacache_value,  # use_teacache
            seed_value,  # seed
            cfg_value,  # cfg
            rs_value,  # rs
            gpu_memory_value,  # gpu_memory
            mp4_crf_value,  # mp4_crf
            keep_temp_png_value,  # keep_temp_png
            keep_temp_json_value  # keep_temp_json
        )

def delete_quick_prompt(prompt_text):
    global quick_prompts
    if prompt_text:
        quick_prompts = [item for item in quick_prompts if item['prompt'] != prompt_text]
        with open(QUICK_LIST_FILE, 'w') as f:
            json.dump(quick_prompts, f, indent=2)
    # Clear the prompt box and quick list selection
    return "", "", gr.update(choices=[item['prompt'] for item in quick_prompts], value=None), 5.0, 10.0, 25, True, -1, 1.0, 0.0, 6.0, 16

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

def convert_model_path(path):
    """Convert from directory name format to huggingface format"""
    # Remove any default model suffixes if present
    if " - USERS DEFAULT MODEL" in path:
        path = path.replace(" - USERS DEFAULT MODEL", "")
    if " - ORIGINAL DEFAULT MODEL" in path:
        path = path.replace(" - ORIGINAL DEFAULT MODEL", "")
    
    # First check if this is a display name with our prefix
    if path.startswith('DOWNLOADED-MODEL-'):
        # Get the actual folder name from our mapping
        if hasattr(Config, 'model_name_mapping') and path in Config.model_name_mapping:
            path = Config.model_name_mapping[path]
    
    # Then do the normal conversion
    if path.startswith('models--'):
        # Convert from "models--org--model" to "org/model"
        parts = path.split('--')
        if len(parts) >= 3:
            return f"{parts[1]}/{parts[2]}"
    return path

# Update model loading code
text_encoder = LlamaModel.from_pretrained(convert_model_path(Config.DEFAULT_TEXT_ENCODER), subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained(convert_model_path(Config.DEFAULT_TEXT_ENCODER_2), subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained(convert_model_path(Config.DEFAULT_TOKENIZER), subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained(convert_model_path(Config.DEFAULT_TOKENIZER_2), subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained(convert_model_path(Config.DEFAULT_VAE), subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained(convert_model_path(Config.DEFAULT_FEATURE_EXTRACTOR), subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained(convert_model_path(Config.DEFAULT_IMAGE_ENCODER), subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(convert_model_path(Config.DEFAULT_TRANSFORMER), torch_dtype=torch.bfloat16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

# if args.lora:
    # lora = args.lora
    # lora_path, lora_name = os.path.split(lora)
    # print("Loading lora")
    # transformer = load_lora(transformer, lora_path, lora_name, args.lora_is_diffusers)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

# Move stream creation to module level, before the process and worker functions
stream = None

def initialize_stream():
    global stream
    if stream is None:
        stream = AsyncStream()
    return stream

def clean_up_temp_mp4png(job):
    job_name = job.job_name

    #Deletes all '<job_name>_<n>.mp4' in outputs_folder except the one with the largest n. Also deletes the '<job_name>.png' file and '<job_name>.json' file. Uses the keep_temp settings from the job object to determine which files to keep.
    
    if job.keep_temp_png:
        debug_print(f"Keeping temporary PNG file for job {job_name} as requested")
    if keep_temp_mp4:
        debug_print(f"Keeping temporary MP4 files for job {job_name} as requested")
    if job.keep_temp_json:
        debug_print(f"Keeping temporary JSON file for job {job_name} as requested")

    # Delete the PNG file
    png_path = os.path.join(job.job_history_folder if hasattr(job, 'job_history_folder') else Config.JOB_HISTORY_FOLDER, f'{job_name}.png')
    try:
        if os.path.exists(png_path) and not job.keep_temp_png:
            os.remove(png_path)
            debug_print(f"Deleted PNG file: {png_path}")
    except OSError as e:
        alert_print(f"Failed to delete PNG file {png_path}: {e}")

    # Delete the job_name.JSON job file
    json_path = os.path.join(job.job_history_folder if hasattr(job, 'job_history_folder') else Config.JOB_HISTORY_FOLDER, f'{job_name}.json')
    try:
        if os.path.exists(json_path) and not job.keep_temp_json:
            os.remove(json_path)
            debug_print(f"Deleted JSON file: {json_path}")
    except OSError as e:
        alert_print(f"Failed to delete JSON file {json_path}: {e}")

    # regex to grab the trailing number
    pattern = re.compile(rf'^{re.escape(job_name)}_(\d+)\.mp4$')
    candidates = []
    
    # scan directory
    for fname in os.listdir(Config.OUTPUTS_FOLDER): 
        m = pattern.match(fname)
        if m:
            frame_count = int(m.group(1))
            candidates.append((frame_count, fname))

    if not candidates:
        return  # nothing to clean up

    # find the highest frame‐count
    highest_count, highest_fname = max(candidates, key=lambda x: x[0])

    # delete all but the highest
    for count, fname in candidates:
        if count != highest_count and not (keep_temp_mp4 and fname.endswith('.mp4')):
            path = os.path.join(Config.OUTPUTS_FOLDER, fname)
            try:
                os.remove(path)
            except OSError as e:
                alert_print(f"Failed to delete {fname}: {e}")

    # Rename the remaining MP4 to {job_name}.mp4
    if highest_fname:
        old_path = os.path.join(Config.OUTPUTS_FOLDER, highest_fname)
        new_path = os.path.join(Config.OUTPUTS_FOLDER, f"{job_name}.mp4")
        try:
            if os.path.exists(new_path):
                os.remove(new_path)  # Remove existing file if it exists
            os.rename(old_path, new_path)
            
            # Check if we need to move the file to a custom output folder
            if hasattr(job, 'outputs_folder') and job.outputs_folder != Config.OUTPUTS_FOLDER:
                try:
                    # Create the custom output directory if it doesn't exist
                    os.makedirs(job.outputs_folder, exist_ok=True)
                    custom_path = os.path.join(job.outputs_folder, f"{job_name}.mp4")
                    debug_print(f"Moving completed video to custom output folder: {custom_path}")
                    
                    # If a file already exists at the destination, remove it
                    if os.path.exists(custom_path):
                        os.remove(custom_path)
                        
                    # Move the file to the custom location
                    shutil.move(new_path, custom_path)
                    debug_print(f"Successfully moved video to custom output folder")
                except Exception as e:
                    alert_print(f"Failed to move video to custom output folder {job.outputs_folder}: {e}")
                    alert_print("Video will remain in default output folder")
                    traceback.print_exc()
            
        except OSError as e:
            alert_print(f"Failed to rename {highest_fname} to {job_name}.mp4: {e}")

def mark_job_processing(job):
    #Mark a job as processing and update its thumbnail
    job.status = "processing"
    
    # Delete existing thumbnail if it exists
    if job.thumbnail and os.path.exists(job.thumbnail):
        os.remove(job.thumbnail)

    job.thumbnail = create_thumbnail(job, status_change=True)

    
    # Move job to top of queue
    if job in job_queue:
        job_queue.remove(job)
        job_queue.insert(0, job)
        
    save_queue()
    return update_queue_table(), update_queue_display()


def mark_job_completed(completed_job):
    #Mark a job as completed and update its thumbnail
    completed_job.status = "completed"
    # Move completed_job to the top of completed jobs
    if completed_job in job_queue:
        job_queue.remove(completed_job)
        # Find the first completed job
        insert_index = len(job_queue)
        for i, existing_job in enumerate(job_queue):
            if existing_job.status == "completed":
                insert_index = i
                break
        job_queue.insert(insert_index, completed_job)
    
    if completed_job.image_path == "text2video":
        #  code to Just update the overlay of the text2video completed_job.thumbnail 
        debug_print(f"in mark_job_completed {completed_job.job_name} is a completed {completed_job.image_path} job so we just add a text overlay")
        mp4_path = os.path.join(completed_job.outputs_folder if hasattr(completed_job, 'outputs_folder') else Config.OUTPUTS_FOLDER, f"{completed_job.job_name}.mp4")
        extract_thumb_from_processing_mp4(completed_job, mp4_path)

    else:
        # Delete existing thumbnail if it exists
        if completed_job.thumbnail and os.path.exists(completed_job.thumbnail):
            os.remove(completed_job.thumbnail)
        
        # Create new thumbnail with completed status
        if os.path.exists(completed_job.image_path):
            completed_job.thumbnail = create_thumbnail(completed_job, status_change=True)
    save_queue()
    return update_queue_table(), update_queue_display()

def mark_job_failed(job):
    #Mark a job as completed and update its thumbnail
    try:
        job.status = "failed"
        
        # Delete existing thumbnail if it exists
        if job.thumbnail and os.path.exists(job.thumbnail):
            os.remove(job.thumbnail)
        
        # Create new thumbnail with failed status
        if job.image_path:
            job.thumbnail = create_thumbnail(job, status_change=True)
        
        # Move job to top of queue
        if job in job_queue:
            job_queue.remove(job)
            job_queue.insert(0, job)
            
        save_queue()
        return update_queue_table(), update_queue_display()
            
    except Exception as e:
        alert_print(f"Error marking job as failed: {str(e)}")
        traceback.print_exc()
        return gr.update(), gr.update()

def mark_job_pending(job):
    #Mark a job as pending and update its thumbnail
    try:
        job.status = "pending"
        
        # Delete existing thumbnail if it exists
        if job.thumbnail and os.path.exists(job.thumbnail):
            os.remove(job.thumbnail)
        
        # Create new clean thumbnail
        if job.image_path:
            job.thumbnail = create_thumbnail(job, status_change=True)
            
        save_queue()
        return update_queue_table(), update_queue_display()
    except Exception as e:
        alert_print(f"Error marking job as pending: {str(e)}")
        traceback.print_exc()
        return gr.update(), gr.update()

@torch.no_grad()
def worker(next_job):
    """Worker function to process a job"""
    global stream
    # Create job output and history folders if they don't exist
    try:
        if not os.path.exists(next_job.job_history_folder):
            debug_print(f"Creating job history folder: {next_job.job_history_folder}") 
            os.makedirs(next_job.job_history_folder, exist_ok=True)
    except Exception as e:
        alert_print(f"Error creating job folders: {str(e)}")
        traceback.print_exc()
        raise

    debug_print(f"Starting worker for job {next_job.job_name}")
    
    # prepare input_image & Handle text-to-video case
    if next_job.image_path == "text2video":
        worker_input_image = None
    else:
        try:
            worker_input_image = np.array(Image.open(next_job.image_path))
        except Exception as e:
            alert_print(f"ERROR loading image: {str(e)}")
            traceback.print_exc()
            raise

    checked_seed = next_job.seed if hasattr(next_job, 'seed') else Config.DEFAULT_SEED
    debug_print(f"Job {next_job.job_name} initial seed value: {checked_seed}")
    
    # Generate random seed if seed is -1
    if checked_seed == -1:
        checked_seed = random.randint(0, 2**32 - 1)
        debug_print(f"Generated new random seed for job {next_job.job_name}: {checked_seed}")
    else:
        checked_seed = checked_seed

    # Extract all values from next_job at the start
    worker_prompt = next_job.prompt if hasattr(next_job, 'prompt') else None
    worker_n_prompt = next_job.n_prompt if hasattr(next_job, 'n_prompt') else None
    worker_seed = checked_seed
    worker_job_name = next_job.job_name 
    worker_latent_window_size = next_job.latent_window_size if hasattr(next_job, 'latent_window_size') else Config.DEFAULT_LATENT_WINDOW_SIZE
    worker_video_length = next_job.video_length if hasattr(next_job, 'video_length') else Config.DEFAULT_VIDEO_LENGTH
    worker_steps = next_job.steps if hasattr(next_job, 'steps') else Config.DEFAULT_STEPS
    worker_cfg = next_job.cfg if hasattr(next_job, 'cfg') else Config.DEFAULT_CFG
    worker_gs = next_job.gs if hasattr(next_job, 'gs') else Config.DEFAULT_GS
    worker_rs = next_job.rs if hasattr(next_job, 'rs') else Config.DEFAULT_RS
    worker_gpu_memory = next_job.gpu_memory if hasattr(next_job, 'gpu_memory') else Config.DEFAULT_GPU_MEMORY
    worker_use_teacache = next_job.use_teacache if hasattr(next_job, 'use_teacache') else Config.DEFAULT_USE_TEACACHE
    worker_mp4_crf = next_job.mp4_crf if hasattr(next_job, 'mp4_crf') else Config.DEFAULT_MP4_CRF
    worker_keep_temp_png = next_job.keep_temp_png if hasattr(next_job, 'keep_temp_png') else Config.DEFAULT_KEEP_TEMP_PNG
    worker_keep_temp_json = next_job.keep_temp_json if hasattr(next_job, 'keep_temp_json') else Config.DEFAULT_KEEP_TEMP_JSON
    worker_outputs_folder = next_job.outputs_folder if hasattr(next_job, 'outputs_folder') else Config.OUTPUTS_FOLDER
    worker_job_history_folder = next_job.job_history_folder if hasattr(next_job, 'job_history_folder') else Config.JOB_HISTORY_FOLDER
    worker_keep_temp_mp4 = next_job.keep_temp_mp4 if hasattr(next_job, 'keep_temp_mp4') else Config.KEEP_TEMP_MP4
    worker_keep_completed_job = next_job.keep_completed_job if hasattr(next_job, 'keep_completed_job') else Config.KEEP_COMPLETED_JOB


    if worker_keep_temp_json:
        job_params = {
            'prompt': worker_prompt,
            'n_prompt': worker_n_prompt,
            'seed': worker_seed,
            'job_name': worker_job_name,
            'length': worker_video_length,
            'steps': worker_steps,
            'cfg': worker_cfg,
            'gs': worker_gs,
            'rs': worker_rs,
            'gpu_memory': worker_gpu_memory,
            'use_teacache': worker_use_teacache,
            'mp4_crf': worker_mp4_crf,
        }
        json_path = os.path.join(worker_job_history_folder, f'{worker_job_name}.json')
        with open(json_path, 'w') as f:
            json.dump(job_params, f, indent=2)

    # Save the input image with metadata
    metadata = PngInfo()
    metadata.add_text("prompt", worker_prompt)
    metadata.add_text("job_name", worker_job_name)
    metadata.add_text("n_prompt", worker_n_prompt) 
    metadata.add_text("seed", str(worker_seed))  # This will now be the random seed if it was -1
    metadata.add_text("video_length", str(worker_video_length))
    metadata.add_text("steps", str(worker_steps))
    metadata.add_text("cfg", str(worker_cfg))
    metadata.add_text("gs", str(worker_gs))
    metadata.add_text("rs", str(worker_rs))
    metadata.add_text("gpu_memory", str(worker_gpu_memory))
    metadata.add_text("use_teacache", str(worker_use_teacache))
    metadata.add_text("mp4_crf", str(worker_mp4_crf))

    debug_print(f"Starting worker for job {worker_job_name}")
    debug_print(f"Worker - Initial parameters: video_length={worker_video_length}, steps={worker_steps}, seed={worker_seed}")
    debug_print(f"Worker - Using stream object: {id(stream)}")

    total_latent_sections = (worker_video_length * 30) / (worker_latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    # job_failed = None not used yet
    # job_id = generate_timestamp() #not used yet
    debug_print(f"Worker - Total latent sections to process: {total_latent_sections}")

    stream.output_queue.push(('progress', (None, "Initializing...", make_progress_bar_html(0, "Step Progress"), "Starting job...", make_progress_bar_html(0, "Job Progress"))))
    debug_print("Worker - Initial progress update pushed")
    debug_print("Worker - Progress update pushed to queue")

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        stream.output_queue.push(('progress', (None, "Text encoding...", make_progress_bar_html(0, "Step Progress"), "Processing...", make_progress_bar_html(0, "Job Progress"))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(worker_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if worker_cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(worker_n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image
        stream.output_queue.push(('progress', (None, "Image processing...", make_progress_bar_html(0, "Step Progress"), "Processing...", make_progress_bar_html(0, "Job Progress"))))

        # Handle text-to-video case
        if worker_input_image is None:
            # Create a blank image for text-to-video with default resolution
            default_resolution = 640  # Default resolution for text-to-video
            input_image_np = np.zeros((default_resolution, default_resolution, 3), dtype=np.uint8)
            height = width = default_resolution
        else:
            # Handle image-to-video case
            input_image_np = np.array(worker_input_image)
            H, W, C = input_image_np.shape
            height, width = find_nearest_bucket(H, W, resolution=640)
            input_image_np = resize_and_center_crop(input_image_np, target_width=width, target_height=height)

            Image.fromarray(input_image_np).save(os.path.join(worker_job_history_folder, f'{worker_job_name}.png'), pnginfo=metadata)

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        stream.output_queue.push(('progress', (None, "VAE encoding...", make_progress_bar_html(0, "Step Progress"), "Processing...", make_progress_bar_html(0, "Job Progress"))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision
        stream.output_queue.push(('progress', (None, "CLIP Vision encoding...", make_progress_bar_html(0, "Step Progress"), "Processing...", make_progress_bar_html(0, "Job Progress"))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        stream.output_queue.push(('progress', (None, "Start sampling...", make_progress_bar_html(0, "Step Progress"), "Processing...", make_progress_bar_html(0, "Job Progress"))))

        rnd = torch.Generator("cpu").manual_seed(worker_seed)

        # Initialize history latents with improved structure
        history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1

        for section_index in range(total_latent_sections):
            if stream.input_queue.top() == 'abort':
                debug_print("Worker - Received abort signal, stopping processing")
                stream.output_queue.push(('abort', None))
                return

            print(f'section_index = {section_index}, total_latent_sections = {total_latent_sections}')
            if not high_vram:
                debug_print("Worker - Managing VRAM for next section")
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=worker_gpu_memory)

            if worker_use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=worker_steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)


            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'abort':
                    stream.output_queue.push(('abort', None))
                    raise KeyboardInterrupt('User aborts the task.')
                current_step = d['i'] + 1
                step_percentage = int(100.0 * current_step / worker_steps)
                step_desc = f'Step {current_step} of {worker_steps}'
                step_progress = make_progress_bar_html(step_percentage, f'Step Progress: {step_percentage}%')

                current_time = max(0, (total_generated_latent_frames * 4 - 3) / 30)
                job_percentage = int((current_time / worker_video_length) * 100)
                job_type = "Image to Video" if next_job.image_path != "text2video" else "Text 2 Video"
                job_desc = f'Creating a {job_type} for job name {worker_job_name} , with these values seed: {worker_seed} cfg scale:{worker_gs} teacache:{worker_use_teacache} mp4_crf:{worker_mp4_crf} Created {current_time:.1f} second(s) of the {worker_video_length} second video - ({job_percentage}% complete), it will be saved in {worker_outputs_folder}{worker_job_name}.mp4'
                job_progress = make_progress_bar_html(job_percentage, f'Job Progress: {job_percentage}%')
                
                

                stream.output_queue.push(('progress', (preview, step_desc, step_progress, job_desc, job_progress)))
                return
            indices = torch.arange(0, sum([1, 16, 2, 1, worker_latent_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, worker_latent_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=worker_latent_window_size * 4 - 3,
                real_guidance_scale=worker_cfg,
                distilled_guidance_scale=worker_gs,
                guidance_rescale=worker_rs,
                # shift=3.0,
                num_inference_steps=worker_steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = worker_latent_window_size * 2
                overlapped_frames = worker_latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{worker_job_name}_{total_generated_latent_frames}.mp4')
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=worker_mp4_crf)
            
            # Calculate current progress percentage
            current_time = max(0, (total_generated_latent_frames * 4 - 3) / 30)
            job_percentage = int((current_time / worker_video_length) * 100)
            stream.output_queue.push(('file', (output_filename, job_percentage)))

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

    except:

        traceback.print_exc()
        
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    completed_job = next_job
    stream.output_queue.push(('done', completed_job))

# where was this          mp4_path = output_filename
            
            
def extract_thumb_from_processing_mp4(next_job, output_filename, job_percentage=0):
    mp4_path = output_filename
    status_overlay = "RUNNING" if next_job.status=="processing" else "DONE" if next_job.status=="completed" else next_job.status.upper()
    status_color = {
        "pending": (0, 255, 255),  # BGR for yellow
        "processing": (255, 0, 0),  # BGR for blue
        "completed": (0, 255, 0),   # BGR for green
        "failed": (0, 0, 255)       # BGR for red
    }.get(next_job.status, (255, 255, 255))  # BGR for white

    if os.path.exists(mp4_path):
        import cv2

        cap = cv2.VideoCapture(mp4_path)
        # Seek to the 10th frame (zero-based index 9)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 9)
        ret, frame = cap.read()
        if ret:
            # target thumbnail size
            THUMB_SIZE = 200

            # get frame dims
            h, w = frame.shape[:2]

            # scale so that the larger dimension becomes THUMB_SIZE
            scale = THUMB_SIZE / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # resize the frame
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # create black background and center the resized frame
            thumb = np.zeros((THUMB_SIZE, THUMB_SIZE, 3), dtype=np.uint8)
            x_off = (THUMB_SIZE - new_w) // 2
            y_off = (THUMB_SIZE - new_h) // 2
            thumb[y_off : y_off + new_h, x_off : x_off + new_w] = resized

            # Overlay centered status text, 
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1
            thickness = 2

            if next_job.status == "processing":
                text1 = f"{job_percentage}%"
                text2 = "RUNNING"
                
                # Calculate text sizes for both lines
                (text1_w, text1_h), _ = cv2.getTextSize(text1, font, scale, thickness)
                (text2_w, text2_h), _ = cv2.getTextSize(text2, font, scale, thickness)
                
                x1 = (thumb.shape[1] - text1_w) // 2
                x2 = (thumb.shape[1] - text2_w) // 2
                y1 = (thumb.shape[0] - text2_h) // 2  # Center vertically
                y2 = y1 + text2_h + 10  # Add small gap between lines
                
                # Add black outline and text for both lines
                for offset in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                    cv2.putText(thumb, text1, (x1+offset[0], y1+offset[1]), font, scale, (0,0,0), thickness+2, cv2.LINE_AA)
                    cv2.putText(thumb, text2, (x2+offset[0], y2+offset[1]), font, scale, (0,0,0), thickness+2, cv2.LINE_AA)
                
                # Add colored text
                cv2.putText(thumb, text1, (x1, y1), font, scale, status_color, thickness, cv2.LINE_AA)
                cv2.putText(thumb, text2, (x2, y2), font, scale, status_color, thickness, cv2.LINE_AA)
            else:
                # Single line for other statuses
                text = status_overlay
                (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
                x = (thumb.shape[1] - text_w) // 2
                y = (thumb.shape[0] + text_h) // 2
                
                # Add black outline
                for offset in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                    cv2.putText(thumb, text, (x+offset[0], y+offset[1]), font, scale, (0,0,0), thickness+2, cv2.LINE_AA)
                
                # Add colored text
                cv2.putText(thumb, text, (x, y), font, scale, status_color, thickness, cv2.LINE_AA)

            thumb_path = os.path.join(temp_queue_images, f'thumb_{next_job.job_name}.png')
            cv2.imwrite(thumb_path, thumb)
        cap.release()
    return(next_job, output_filename)




def process():
    global stream
    stream = initialize_stream()
    
    # First check for pending jobs
    pending_jobs = [job for job in job_queue if job.status.lower() == "pending"]

    if not pending_jobs:
        # No pending jobs
        yield (
            gr.update(interactive=True),      # queue_button
            gr.update(interactive=True),      # start_button
            gr.update(interactive=False),     # abort_button
            None,                             # preview_image
            gr.update(visible=False),         # result_video
            "",                               # progress_desc1
            "",                               # progress_bar1
            "no pending jobs to process",     # progress_desc2
            "",                               # progress_bar2
            update_queue_display(),           # queue_display
            update_queue_table()              # queue_table
        )
        return

    # Process first pending job
    pending_job = pending_jobs[0]
    queue_table_update, queue_display_update = mark_job_processing(pending_job)
    save_queue()
    
    # Start processing
    debug_print(f"Starting worker for job {pending_job.job_name}")
    
    async_run(worker, pending_job)    ###### the first run is needed to start the stream all later runs will be dunt in the while true loop. pending job is the one that will be processed

    # Initial yield - Fixed to include queue_table
    yield (
        gr.update(interactive=True),      # queue_button
        gr.update(interactive=False),     # start_button
        gr.update(interactive=True),      # abort_button
        gr.update(visible=True),          # preview_image
        gr.update(value=None),          # result_video
        "",                               # progress_desc1
        "",                               # progress_bar1
        "Starting job processing...",     # progress_desc2
        "",                               # progress_bar2
        update_queue_display(),           # queue_display
        update_queue_table()              # queue_table
    )

    # Process output queue
    while True:
        try:
            flag, data = stream.output_queue.next()
            #debug_print(f"Process - After stream.output_queue.next(), got flag: {flag}")

            if flag == 'file':
                debug_print("[DEBUG] Process - Handling file flag")
                output_filename, job_percentage = data
                
                
                processing_jobs = [job for job in job_queue if job.status.lower() == "processing"]
                if processing_jobs:
                    file_job = processing_jobs[0]
                
                
                # Ensure path is absolute
                if not os.path.isabs(output_filename):
                    output_filename = os.path.abspath(output_filename)
                


                extract_thumb_from_processing_mp4(file_job, output_filename, job_percentage)
                

                
                yield (
                    gr.update(interactive=True),    # queue_button
                    gr.update(interactive=False),   # start_button
                    gr.update(interactive=True),    # abort_button
                    gr.update(visible=True),        # preview_image (File Output: visible)
                    gr.update(output_filename),  # result_video
                    "",    # keep last step progress
                    "",       # keep last step progress bar
                    "",     # keep last job progress
                    "",        # keep last job progress bar
                    update_queue_display(),         # queue_display
                    update_queue_table()           # queue_table
                )
                
          #=======================PROGRESS STAGE=============

            if flag == 'progress':
                preview, step_desc, step_progress, job_desc, job_progress = data
                yield (
                    gr.update(interactive=True),    # queue_button
                    gr.update(interactive=False),   # start_button
                    gr.update(interactive=True),    # abort_button
                    gr.update(visible=True, value=preview), # preview_image
                    gr.update(),                    # leave result_video as is
                    step_desc,                      # progress_desc1
                    step_progress,                  # progress_bar1 
                    job_desc, #causes flicker ?                     # progress_desc2 
                    job_progress,                   # progress_bar2 
                    update_queue_display(),         # queue_display
                    update_queue_table()           # queue_table
                )


            if flag == 'abort':
                if stream.input_queue.top() == 'abort':
                    aborted_job = next((job for job in job_queue if job.status == "processing"), None)
                    if aborted_job:
                        clean_up_temp_mp4png(aborted_job)
                        alert_print(f"trying to save aborted job to outputs folder {aborted_job.outputs_folder}")
                        mp4_path = os.path.join(aborted_job.outputs_folder if hasattr(aborted_job, 'outputs_folder') else Config.OUTPUTS_FOLDER, f"{aborted_job.job_name}.mp4") 
                        extract_thumb_from_processing_mp4(aborted_job, mp4_path)
                        debug_print(f"aborted job video saved to outputs folder {aborted_job.outputs_folder}")
                        queue_table_update, queue_display_update = mark_job_pending(aborted_job)
                        save_queue()
                        
                        
                        yield (
                            gr.update(interactive=True),    # queue_button
                            gr.update(interactive=True),    # start_button
                            gr.update(interactive=False),   # abort_button
                            gr.update(visible=False),       # preview_image (Abort: hidden)
                            gr.update(visible=False),       # result_video (Abort: hidden)
                            "Job Aborted",                  # progress_desc1 (step progress)

                            "",                            # progress_bar1 (step progress)
                            "Processing stopped",          # progress_desc2 (job progress)
                            "",                            # progress_bar2 (job progress)
                            update_queue_display(),         # queue_display
                            update_queue_table()           # queue_table
                        )

                        return


            if flag == 'done':
                completed_job = data
                print(f"completed job recieved at done flag job name {completed_job.job_name}")


                # previous job completed
                clean_up_temp_mp4png(completed_job)
                debug_print(f"extracting thumb from completed job {completed_job.job_name}.mp4 to the outputs folder {completed_job.outputs_folder}")
                mp4_path = os.path.join(completed_job.outputs_folder if hasattr(completed_job, 'outputs_folder') else Config.OUTPUTS_FOLDER, f"{completed_job.job_name}.mp4")
                extract_thumb_from_processing_mp4(completed_job, mp4_path)
                mark_job_completed(completed_job)
                save_queue()

                # Check for next pending job
                next_job = None
                pending_jobs = [job for job in job_queue if job.status.lower() == "pending"]
                if pending_jobs:
                    next_job = pending_jobs[0]
                if next_job:
                    debug_print(f"now marking next job {next_job.job_name} as processing and preparing to send to worker")    
                    queue_table_update, queue_display_update = mark_job_processing(next_job)
                    save_queue()
                    yield (
                        gr.update(interactive=True),    # queue_button
                        gr.update(interactive=False),    # start_button
                        gr.update(interactive=True),   # abort_button
                        gr.update(visible=True),       # preview_image 
                        gr.update(output_filename),  # show result_video with final file
                        "Generation Complete",          # progress_desc1 (step progress)
                        make_progress_bar_html(100, "Complete"),  # progress_bar1 (step progress)
                        "Job Finished Successfully",    # progress_desc2 (job progress)
                        make_progress_bar_html(100, "Complete"),  # progress_bar2 (job progress)
                        update_queue_display(),         # queue_display
                        update_queue_table()           # queue_table
                    )
                        

                    debug_print(f"Starting worker for job {next_job.job_name}")
                    async_run(worker, next_job)
                    yield (
                        gr.update(interactive=True),      # queue_button
                        gr.update(interactive=False),     # start_button
                        gr.update(interactive=True),      # abort_button
                        gr.update(visible=True),          # preview_image
                        gr.update(value=None),            # result_video
                        "Initializing steps...",          # progress_desc1 (step progress)
                        make_progress_bar_html(0, "Preparing"),  # progress_bar1 (step progress)
                        "Starting job processing...",     # progress_desc2 (job progress)
                        make_progress_bar_html(0, "Job Progress"),  # progress_bar2 (job progress)
                        update_queue_display(),           # queue_display
                        update_queue_table()             # queue_table
                    )

                else:
                    debug_print("No more pending jobs to process")
                    yield (
                        gr.update(interactive=True),   # queue_button (always enabled)
                        gr.update(interactive=True),   # start_button
                        gr.update(interactive=False),  # abort_button
                        None,  # preview_image
                        gr.update(output_filename),  # show result_video with final file
                        "No more pending jobs to process",  # progress_desc1 (step progress)
                        make_progress_bar_html(100, "Complete"),  # progress_bar1 (step progress)
                        "Job Finished Successfully",    # progress_desc2 (job progress)
                        make_progress_bar_html(100, "Complete"),  # progress_bar2 (job progress)
                        update_queue_display(),        # queue_display
                        update_queue_table()         # queue_table
                    )

                    return

        except Exception as e:
            debug_print(f"Error in process loop: {str(e)}")
            return

def end_process():
    """Handle abort generation button click - stop all processes and change all processing jobs to pending jobs"""
    stream.input_queue.push('abort')
    return (
        update_queue_table(),  # dataframe
        update_queue_display(),  # gallery
        gr.update(interactive=True),  # queue_button
        gr.update(interactive=True),  # start_button
        gr.update(interactive=False)  # abort_button
    )
    

def add_to_queue_handler(input_image, prompt, n_prompt, video_length, seed, job_name, use_teacache, gpu_memory, steps, cfg, gs, rs, mp4_crf, keep_temp_png, keep_temp_json, create_job_outputs_folder=None, create_job_history_folder=None, keep_temp_mp4=None, create_job_keep_completed_job=None):
    """Handle adding a new job to the queue"""
    try:
        debug_print("=== add_to_queue_handler input values ===")
        debug_print(f"input_image type: {type(input_image)}")
        debug_print(f"prompt type: {type(prompt)}")
        debug_print(f"video_length type: {type(video_length)}")
        debug_print(f"seed type: {type(seed)}")
        debug_print(f"job_name type: {type(job_name)}")
        debug_print(f"use_teacache type: {type(use_teacache)}")
        debug_print(f"gpu_memory type: {type(gpu_memory)}")
        debug_print(f"steps type: {type(steps)}")
        debug_print(f"cfg type: {type(cfg)}")
        debug_print(f"gs type: {type(gs)}")
        debug_print(f"rs type: {type(rs)}")
        debug_print(f"mp4_crf type: {type(mp4_crf)}")
        debug_print(f"keep_temp_png type: {type(keep_temp_png)}")
        debug_print(f"keep_temp_json type: {type(keep_temp_json)}")
        debug_print(f"create_job_outputs_folder type: {type(create_job_outputs_folder)}")
        debug_print(f"create_job_history_folder type: {type(create_job_history_folder)}")
        debug_print(f"keep_temp_mp4 type: {type(keep_temp_mp4)}")
        debug_print(f"create_job_keep_completed_job type: {type(create_job_keep_completed_job)}")
        debug_print("=====================================")

        if prompt is None and input_image is None:
            return (
                update_queue_table(),         # queue_table
                update_queue_display(),       # queue_display
                gr.update(interactive=True)   # queue_button (always enabled)
            )
        if not job_name:  # This will catch both None and empty string
            job_name = "job"  # Remove the underscore here since we add it later
        # Handle text-to-video case (no input image)
        if input_image is None:
            job_name = add_to_queue(
                prompt=prompt,
                n_prompt=n_prompt,
                input_image=None,  # Pass None for text-to-video
                video_length=video_length,
                seed=seed,
                job_name=job_name,
                use_teacache=use_teacache,
                gpu_memory=gpu_memory,
                steps=steps,
                cfg=cfg,
                gs=gs,
                rs=rs,
                status="pending",
                mp4_crf=mp4_crf,
                keep_temp_png=keep_temp_png,
                keep_temp_json=keep_temp_json,
                create_job_outputs_folder=create_job_outputs_folder,
                create_job_history_folder=create_job_history_folder,
                keep_temp_mp4=keep_temp_mp4,
                create_job_keep_completed_job=create_job_keep_completed_job
            )
            save_queue()
            return (
                update_queue_table(),         # queue_table
                update_queue_display(),       # queue_display
                gr.update(interactive=True)   # queue_button (always enabled)
            )

        # Handle image-to-video cases
        if isinstance(input_image, list):
            # Multiple images case
            original_job_name = job_name  # Store the original job name prefix
            for img_tuple in input_image:
                input_image = np.array(Image.open(img_tuple[0]))  # Convert to numpy array
                
                # Add job for each image, using original job name prefix
                job_name = add_to_queue(
                    prompt=prompt,
                    n_prompt=n_prompt,
                    input_image=input_image,
                    video_length=video_length,
                    seed=seed,
                    job_name=original_job_name,  # Use original prefix each time
                    use_teacache=use_teacache,
                    gpu_memory=gpu_memory,
                    steps=steps,
                    cfg=cfg,
                    gs=gs,
                    rs=rs,
                    status="pending",
                    mp4_crf=mp4_crf,
                    keep_temp_png=keep_temp_png,
                    keep_temp_json=keep_temp_json,
                    create_job_outputs_folder=create_job_outputs_folder,
                    create_job_history_folder=create_job_history_folder,
                    keep_temp_mp4=keep_temp_mp4,
                    create_job_keep_completed_job=create_job_keep_completed_job
                )
                # Create thumbnail for the job
                job = next((job for job in job_queue if job.job_name == job_name), None)
                if job and job.image_path:
                    job.thumbnail = create_thumbnail(job, status_change=True)
                    save_queue()
        else:
            # Single image case
            input_image = np.array(Image.open(input_image[0]))  # Convert to numpy array
            
            # Add single image job
            job_name = add_to_queue(
                prompt=prompt,
                n_prompt=n_prompt,
                input_image=input_image,
                video_length=video_length,
                seed=seed,
                job_name=job_name,
                use_teacache=use_teacache,
                gpu_memory=gpu_memory,
                steps=steps,
                cfg=cfg,
                gs=gs,
                rs=rs,
                status="pending",
                mp4_crf=mp4_crf,
                keep_temp_png=keep_temp_png,
                keep_temp_json=keep_temp_json,
                create_job_outputs_folder=create_job_outputs_folder,
                create_job_history_folder=create_job_history_folder,
                keep_temp_mp4=keep_temp_mp4,
                create_job_keep_completed_job=create_job_keep_completed_job
            )
        
            job = next((job for job in job_queue if job.job_name == job_name), None)
            if job and job.image_path:
                job.thumbnail = create_thumbnail(job, status_change=True)
                save_queue()  # Save after changing statuses

        return (
            update_queue_table(),         # queue_table
            update_queue_display(),       # queue_display
            gr.update(interactive=True)   # queue_button (always enabled)
        )
    except Exception as e:
        alert_print(f"Error in add_to_queue_handler: {str(e)}")
        traceback.print_exc()
        return (
            update_queue_table(),         # queue_table
            update_queue_display(),       # queue_display
            gr.update(interactive=True)   # queue_button (always enabled)
        )

def delete_all_jobs():
    """Delete all jobs from the queue and their associated files"""
    try:
        # Delete all job files
        for job in job_queue:
            if os.path.exists(job.image_path):
                os.remove(job.image_path)
            if os.path.exists(job.thumbnail):
                os.remove(job.thumbnail)
        
        # Clear the queue
        job_queue.clear()
        save_queue()
        return update_queue_display()
    except Exception as e:
        alert_print(f"Error deleting all jobs: {str(e)}")
        traceback.print_exc()
        return update_queue_display()

def move_job_to_top(job_name):
    save_queue()

    """Move a job to the top of the queue, maintaining processing job at top"""
    try:
        # Find the job's current index
        current_index = None
        for i, job in enumerate(job_queue):
            if job.job_name == job_name:
                current_index = i
                break
        
        if current_index is None:
            return update_queue_table(), update_queue_display()
        
        # Get the job
        job = job_queue[current_index]
        
        # Remove from current position
        job_queue.pop(current_index)
        
        # Find the first non-processing job
        insert_index = 0
        for i, existing_job in enumerate(job_queue):
            if existing_job.status != "processing":
                insert_index = i
                break
        
        # Insert the job at the found index
        job_queue.insert(insert_index, job)
        save_queue()
        
        return update_queue_table(), update_queue_display()
    except Exception as e:
        alert_print(f"Error moving job to top: {str(e)}")
        traceback.print_exc()
        return update_queue_table(), update_queue_display()

def move_job_to_bottom(job_name):
    save_queue()

    """Move a job to the bottom of the queue, maintaining completed jobs at bottom"""
    try:
        # Find the job's current index
        current_index = None
        for i, job in enumerate(job_queue):
            if job.job_name == job_name:
                current_index = i
                break
        
        if current_index is None:
            return update_queue_table(), update_queue_display()
        
        # Get the job
        job = job_queue[current_index]
        
        # Remove from current position
        job_queue.pop(current_index)
        
        # Find the first completed job
        insert_index = len(job_queue)
        for i, existing_job in enumerate(job_queue):
            if existing_job.status == "completed":
                insert_index = i
                break
        
        # Insert the job at the found index
        job_queue.insert(insert_index, job)
        save_queue()
        
        return update_queue_table(), update_queue_display()
    except Exception as e:
        alert_print(f"Error moving job to bottom: {str(e)}")
        traceback.print_exc()
        return update_queue_table(), update_queue_display()

def move_job(job_name, direction):
    save_queue()

    """Move a job up or down one position in the queue while maintaining sorting rules"""
    try:
        # Find the job's current index
        current_index = None
        for i, job in enumerate(job_queue):
            if job.job_name == job_name:
                current_index = i
                break
        
        if current_index is None:
            return update_queue_table(), update_queue_display()
        
        # Get the job
        job = job_queue[current_index]
        
        # Calculate new index based on direction and sorting rules
        if direction == 'up':
            # Find the previous non-processing job
            new_index = current_index - 1
            while new_index >= 0 and job_queue[new_index].status == "processing":
                new_index -= 1
            if new_index < 0:
                return update_queue_table(), update_queue_display()
            
            # Don't allow moving above processing jobs
            if job.status == "pending" and job_queue[new_index].status == "processing":
                return update_queue_table(), update_queue_display()
                
        else:  # direction == 'down'
            # Find the next non-completed job
            new_index = current_index + 1
            while new_index < len(job_queue) and job_queue[new_index].status == "completed":
                new_index += 1
            if new_index >= len(job_queue):
                return update_queue_table(), update_queue_display()
            
            # Don't allow moving below completed jobs
            if job.status == "pending" and job_queue[new_index].status == "completed":
                return update_queue_table(), update_queue_display()
        
        # Remove from current position and insert at new position
        job_queue.pop(current_index)
        job_queue.insert(new_index, job)
        save_queue()
        
        return update_queue_table(), update_queue_display()
    except Exception as e:
        alert_print(f"Error moving job: {str(e)}")
        traceback.print_exc()
        return update_queue_table(), update_queue_display()

def remove_job(job_name):
    """Delete a job from the queue and its associated files"""
    try:
        # Find and remove job from queue
        for job in job_queue:
            if job.job_name == job_name:
                # Delete associated files
                if job.image_path and os.path.exists(job.image_path):
                    os.remove(job.image_path)
                if os.path.exists(job.thumbnail):
                    os.remove(job.thumbnail)
                job_queue.remove(job)
                break
        save_queue()
        debug_print(f"Total jobs in the queue:{len(job_queue)}")
        return update_queue_table(), update_queue_display()
    except Exception as e:
        alert_print(f"Error deleting job: {str(e)}")
        traceback.print_exc()
        return update_queue_table(), update_queue_display()

def handle_queue_action(evt: gr.SelectData):
    """Handle queue action button clicks"""
    # Default empty return values
    empty_values = (
        "",  # prompt
        "",  # n_prompt
        None,  # video_length
        None,  # seed
        None,  # use_teacache
        None,  # gpu_memory
        None,  # steps
        None,  # cfg
        None,  # gs
        None,  # rs
        None,  # mp4_crf
        None,  # keep_temp_png
        None,  # keep_temp_json
        "",  # outputs_folder
        "",  # job_history_folder
        None,  # keep_temp_mp4
        None,  # keep_completed_job
        "",  # job_name
        gr.update(visible=False)  # edit group visibility
    )

    if evt.index is None or evt.value not in ["⏫️", "⬆️", "⬇️", "⏬️", "❌", "📋", "✎"]:
        return empty_values
    
    row_index, col_index = evt.index
    button_clicked = evt.value
    job = job_queue[row_index]
    
    # Handle actions that don't need job values
    if button_clicked == "⏫️":
        move_job_to_top(job.job_name)
        return empty_values
    elif button_clicked == "⬆️":
        move_job(job.job_name, 'up')
        return empty_values
    elif button_clicked == "⬇️":
        move_job(job.job_name, 'down')
        return empty_values
    elif button_clicked == "⏬️":
        move_job_to_bottom(job.job_name)
        return empty_values
    elif button_clicked == "❌":
        remove_job(job.job_name)
        return empty_values
    
    # Handle actions that need job values
    elif button_clicked == "✎":
        if job.status in ["pending", "completed"]:
            return (
                job.prompt,
                job.n_prompt,
                job.video_length,
                job.seed,
                job.use_teacache,
                job.gpu_memory,
                job.steps,
                job.cfg,
                job.gs,
                job.rs,
                job.mp4_crf,
                job.keep_temp_png,
                job.keep_temp_json,
                job.outputs_folder,
                job.job_history_folder,
                job.keep_temp_mp4,
                job.keep_completed_job,
                job.job_name,
                gr.update(visible=True)
            )
    elif button_clicked == "📋":
        copy_job(job.job_name)
        # After copying, show the original job's values in the edit form
        return (
            job.prompt,
            job.n_prompt,
            job.video_length,
            job.seed,
            job.use_teacache,
            job.gpu_memory,
            job.steps,
            job.cfg,
            job.gs,
            job.rs,
            job.mp4_crf,
            job.keep_temp_png,
            job.keep_temp_json,
            job.outputs_folder,
            job.job_history_folder,
            job.keep_temp_mp4,
            job.keep_completed_job,
            "",  # Clear job_name for the copy
            gr.update(visible=True)
        )
    
    return empty_values

def copy_job(job_name):
    """Create a copy of a job and insert it below the original"""
    try:
        # Find the job
        original_job = next((j for j in job_queue if j.job_name == job_name), None)
        if not original_job:
            return update_queue_table(), update_queue_display()
            
        # Create a new job ID by keeping the prefix and adding a new hex suffix
        prefix = original_job.job_name.rsplit('_', 1)[0]  # Get everything before the last underscore
        new_job_name = f"{prefix}_{uuid.uuid4().hex[:8]}"
        
        # Copy the image file
        if os.path.exists(original_job.image_path):
            new_image_path = os.path.join(temp_queue_images, f"queue_image_{new_job_name}.png")
            shutil.copy2(original_job.image_path, new_image_path)
        else:
            new_image_path = "text2video"
            
        # Create new job with copied parameters
        new_job = QueuedJob(
            prompt=original_job.prompt,
            image_path=new_image_path,
            video_length=original_job.video_length,
            job_name=new_job_name,
            seed=original_job.seed,
            use_teacache=original_job.use_teacache,
            gpu_memory=original_job.gpu_memory,
            steps=original_job.steps,
            cfg=original_job.cfg,
            gs=original_job.gs,
            rs=original_job.rs,
            n_prompt=original_job.n_prompt,
            status="pending",
            thumbnail="",
            mp4_crf=original_job.mp4_crf,
            keep_temp_png=original_job.keep_temp_png,
            keep_temp_json=original_job.keep_temp_json, 
            outputs_folder=original_job.outputs_folder,
            job_history_folder=original_job.job_history_folder,
            keep_temp_mp4=original_job.keep_temp_mp4,
            keep_completed_job=original_job.keep_completed_job
        )
        
        # Find the original job's index
        original_index = job_queue.index(original_job)
        
        # Insert the new job right after the original
        job_queue.insert(original_index + 1, new_job)
        save_queue()
        # Create thumbnail for the new job
        if new_image_path:
            new_job.thumbnail = create_thumbnail(new_job, status_change=True)
       
        # Save the updated queue
        save_queue()
        debug_print(f"Total jobs in the queue:{len(job_queue)}")
        
        return update_queue_table(), update_queue_display()
        
    except Exception as e:
        alert_print(f"Error copying job: {str(e)}")
        traceback.print_exc()
        return update_queue_table(), update_queue_display()

css = make_progress_bar_css() + """
.gradio-gallery-container {
    max-height: 600px !important;
    overflow-y: auto !important;
    padding: 10px;
}
.gradio-gallery-container::-webkit-scrollbar {
    width: 8px !important;
}
.gradio-gallery-container::-webkit-scrollbar-track {
    background: #f0f0f0 !important;
}
.gradio-gallery-container::-webkit-scrollbar-thumb {
    background-color: #666 !important;
    border-radius: 4px !important;
}
.input-gallery,
.queue-gallery {
    max-height: 600px !important;
    overflow-y: auto !important;
    padding: 10px;
}
.input-gallery > div,
.queue-gallery > div {
    height: 100% !important;
    overflow-y: auto !important;
}
.input-gallery .gallery-container,
.queue-gallery .gallery-container {
    max-height: 600px !important;
    overflow-y: auto !important;
    padding: 10px;
}

/* Hide DataFrame headers (remove column numbers) */
.gradio-dataframe thead {
    display: none !important;
}

/* Fix widths for action columns */
.gradio-dataframe th:nth-child(2),
.gradio-dataframe td:nth-child(2),
.gradio-dataframe th:nth-child(3),
.gradio-dataframe td:nth-child(3),
.gradio-dataframe th:nth-child(4),
.gradio-dataframe td:nth-child(4),
.gradio-dataframe th:nth-child(5),
.gradio-dataframe td:nth-child(5),
.gradio-dataframe th:nth-child(6),
.gradio-dataframe td:nth-child(6) {
    width: 42px !important;
    min-width: 42px !important;
    max-width: 42px !important;
    text-align: center !important;
    padding: 0 !important;
}

/* Fix width for prompt column */
.gradio-dataframe th:last-child,
.gradio-dataframe td:last-child {
    width: 300px !important;
    min-width: 200px !important;
    max-width: 400px !important;
    text-align: left !important;
    white-space: normal !important;
    word-break: break-word !important;
}

/* Restore DataFrame child column widths for prompt and others */
.gradio-dataframe th:nth-child(7) { width:  80px !important; min-width:  80px !important; }
.gradio-dataframe th:nth-child(8) { width:  60px !important; min-width:  60px !important; }
.gradio-dataframe th:nth-child(9) { width: 300px !important; min-width: 300px !important; text-align: left !important; }
.gradio-dataframe th:nth-child(10){ width:  60px !important; min-width:  60px !important; }
"""


def edit_job(job_name, new_prompt, new_n_prompt, new_video_length, new_seed, new_use_teacache, new_gpu_memory, new_steps, new_cfg, new_gs, new_rs, new_mp4_crf, new_keep_temp_png, new_keep_temp_json, new_outputs_folder, new_job_history_folder, new_keep_temp_mp4, new_keep_completed_job):
    """Edit a job's parameters"""
    try:
        # Find the job
        for job in job_queue:
            if job.job_name == job_name:
                # Only allow editing if job is pending or completed
                if job.status not in ("pending", "completed"):
                    return update_queue_table(), update_queue_display(), gr.update(visible=False)

                # Update job parameters
                job.prompt = new_prompt
                job.n_prompt = new_n_prompt
                job.video_length = new_video_length
                job.seed = new_seed
                job.use_teacache = new_use_teacache
                job.gpu_memory = new_gpu_memory
                job.steps = new_steps
                job.cfg = new_cfg
                job.gs = new_gs
                job.rs = new_rs
                job.mp4_crf = new_mp4_crf
                job.keep_temp_png = new_keep_temp_png
                job.keep_temp_json = new_keep_temp_json
                job.outputs_folder = new_outputs_folder
                job.job_history_folder = new_job_history_folder
                job.keep_temp_mp4 = new_keep_temp_mp4
                job.keep_completed_job = new_keep_completed_job
            
                # If job was completed, change to pending and move it
                if job.status == "completed":
                    job.status = "pending"
                    # Remove from current position
                    job_queue.remove(job)
                    # Find the first pending job position
                    insert_index = 0
                    for i, existing_job in enumerate(job_queue):
                        if existing_job.status == "pending":
                            insert_index = i
                            break
                    # Insert at the found index
                    job_queue.insert(insert_index, job)
                    create_thumbnail(job, status_change=True)
                    mark_job_pending(job)
                
                # Save changes
                save_queue()
                break
        
        return update_queue_table(), update_queue_display(), gr.update(visible=False)  # Hide edit group
    except Exception as e:
        alert_print(f"Error editing job: {str(e)}")
        traceback.print_exc()
        return update_queue_table(), update_queue_display(), gr.update(visible=False)  # Hide edit group

def delete_completed_jobs():
    """Delete all completed jobs from the queue"""
    global job_queue
    job_queue = [job for job in job_queue if job.status != "completed"]
    save_queue()
    debug_print(f"Total jobs in the queue:{len(job_queue)}")

    return update_queue_table(), update_queue_display()

def delete_pending_jobs():
    """Delete all pending jobs from the queue"""
    global job_queue
    job_queue = [job for job in job_queue if job.status != "pending"]
    save_queue()
    debug_print(f"Total jobs in the queue:{len(job_queue)}")
    return update_queue_table(), update_queue_display()


def delete_failed_jobs():
    """Delete all failed jobs from the queue"""
    global job_queue
    job_queue = [job for job in job_queue if job.status != "failed"]
    save_queue()
    debug_print(f"Total jobs in the queue:{len(job_queue)}")
    return update_queue_table(), update_queue_display()


def hide_edit_window():
    """Hide the edit window without saving changes"""
    return gr.update(visible=False)



def save_system_settings(new_outputs_folder, new_job_history_folder, new_debug_mode, new_keep_temp_mp4, new_keep_completed_job):
    """Save system settings to settings.ini and update runtime values"""
    try:
        # Declare globals first
        global job_history_folder, outputs_folder, debug_mode, keep_temp_mp4, keep_completed_job
        
        config = load_settings()
        
        if 'System Defaults' not in config:
            config['System Defaults'] = {}
            
        # Update the settings
        config['System Defaults']['OUTPUTS_FOLDER'] = repr(new_outputs_folder)
        config['System Defaults']['JOB_HISTORY_FOLDER'] = repr(new_job_history_folder)
        config['System Defaults']['DEBUG_MODE'] = repr(new_debug_mode)
        config['System Defaults']['KEEP_TEMP_MP4'] = repr(new_keep_temp_mp4)
        config['System Defaults']['KEEP_COMPLETED_JOB'] = repr(new_keep_completed_job)
        
        # Save to file
        save_settings(config)
        
        # Update Config object
        Config.OUTPUTS_FOLDER = new_outputs_folder
        Config.JOB_HISTORY_FOLDER = new_job_history_folder
        Config.DEBUG_MODE = new_debug_mode
        Config.KEEP_TEMP_MP4 = new_keep_temp_mp4
        Config.KEEP_COMPLETED_JOB = new_keep_completed_job
        
        # Update global variables
        job_history_folder = new_job_history_folder  # Fixed: was incorrectly set to outputs_folder
        outputs_folder = new_outputs_folder
        debug_mode = new_debug_mode
        keep_temp_mp4 = new_keep_temp_mp4
        keep_completed_job = new_keep_completed_job
        
        # Create directories if they don't exist
        os.makedirs(outputs_folder, exist_ok=True)
        os.makedirs(job_history_folder, exist_ok=True)
        
        return "System settings saved and runtime values updated! New directories created if needed."
    except Exception as e:
        return f"Error saving system settings: {str(e)}"

def restore_system_settings():
    """Restore system settings to original defaults"""
    try:
        # Get original defaults
        defaults = Config.get_original_defaults()
        
        # Load current config
        config = load_settings()
        
        if 'System Defaults' not in config:
            config['System Defaults'] = {}
            
        # Update with original values
        config['System Defaults']['OUTPUTS_FOLDER'] = repr(defaults['OUTPUTS_FOLDER'])
        config['System Defaults']['JOB_HISTORY_FOLDER'] = repr(defaults['JOB_HISTORY_FOLDER'])
        config['System Defaults']['DEBUG_MODE'] = repr(defaults['DEBUG_MODE'])
        config['System Defaults']['KEEP_TEMP_MP4'] = repr(defaults['KEEP_TEMP_MP4'])
        config['System Defaults']['KEEP_COMPLETED_JOB'] = repr(defaults['KEEP_COMPLETED_JOB'])
        
        # Save to file
        save_settings(config)
        
        # Update Config object
        Config.OUTPUTS_FOLDER = defaults['OUTPUTS_FOLDER']
        Config.JOB_HISTORY_FOLDER = defaults['JOB_HISTORY_FOLDER']
        Config.DEBUG_MODE = defaults['DEBUG_MODE']
        Config.KEEP_TEMP_MP4 = defaults['KEEP_TEMP_MP4']
        Config.KEEP_COMPLETED_JOB = defaults['KEEP_COMPLETED_JOB']
        
        # Update global variables
        global job_history_folder, outputs_folder, debug_mode, keep_temp_mp4, keep_completed_job
        job_history_folder = defaults['JOB_HISTORY_FOLDER']  # Fixed: was incorrectly set to outputs_folder
        outputs_folder = defaults['OUTPUTS_FOLDER']
        debug_mode = defaults['DEBUG_MODE']
        keep_temp_mp4 = defaults['KEEP_TEMP_MP4']
        keep_completed_job = defaults['KEEP_COMPLETED_JOB']
        
        # Create directories if they don't exist
        os.makedirs(outputs_folder, exist_ok=True)
        os.makedirs(job_history_folder, exist_ok=True)
        
        # Return values to update UI
        return (
            defaults['OUTPUTS_FOLDER'],
            defaults['JOB_HISTORY_FOLDER'],
            defaults['DEBUG_MODE'],
            defaults['KEEP_TEMP_MP4'],
            defaults['KEEP_COMPLETED_JOB'],
            "System settings restored and runtime values updated! New directories created if needed."
        )
    except Exception as e:
        return None, None, None, None, None, f"Error restoring system settings: {str(e)}"




def set_all_models_as_default(transformer, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder):
    """Set all models as default at once"""
    try:
        # Check if all models are downloaded
        models = [transformer, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder]
        model_types = ['transformer', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'vae', 'feature_extractor', 'image_encoder']
        
        for model, model_type in zip(models, model_types):
            if not model.startswith('LOCAL-'):
                return f"Cannot set {model_type} as default - model must be downloaded first", None, None, None, None, None, None, None, None
        
        config = configparser.ConfigParser()
        config.read(INI_FILE)
        
        if 'Model Defaults' not in config:
            config['Model Defaults'] = {}
            
        # Update all models
        success_messages = []
        for model, model_type in zip(models, model_types):
            actual_model = Config.model_name_mapping.get(model, model.replace('LOCAL-', '').replace(' - CURRENT DEFAULT MODEL', ''))
            config['Model Defaults'][f'DEFAULT_{model_type.upper()}'] = repr(actual_model)
            setattr(Config, f'DEFAULT_{model_type.upper()}', actual_model)
            success_messages.append(f"{model_type}: {actual_model}")
            
        # Save to file
        with open(INI_FILE, 'w') as f:
            config.write(f)
            
        # Get updated model lists
        models = get_available_models(include_online=False)
        
        return (
            "All models set as default successfully:\n" + "\n".join(success_messages) + "\n\nRestart required for changes to take effect.",
            models['transformer'],
            models['text_encoder'],
            models['text_encoder_2'],
            models['tokenizer'],
            models['tokenizer_2'],
            models['vae'],
            models['feature_extractor'],
            models['image_encoder']
        )
    except Exception as e:
        return f"Error setting models as default: {str(e)}", None, None, None, None, None, None, None, None

def restore_all_model_defaults():
    """Restore all models to their original defaults at once"""
    try:
        # Get original defaults
        original_defaults = Config.get_original_defaults()
        
        config = configparser.ConfigParser()
        config.read(INI_FILE)
        
        if 'Model Defaults' not in config:
            config['Model Defaults'] = {}
            
        # Update all model defaults
        success_messages = []
        model_types = ['transformer', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'vae', 'feature_extractor', 'image_encoder']
        
        for model_type in model_types:
            original_value = original_defaults[f'DEFAULT_{model_type.upper()}']
            config['Model Defaults'][f'DEFAULT_{model_type.upper()}'] = repr(original_value)
            setattr(Config, f'DEFAULT_{model_type.upper()}', original_value)
            success_messages.append(f"{model_type}: {original_value}")
            
        # Save to file
        with open(INI_FILE, 'w') as f:
            config.write(f)
            
        # Get updated model lists
        models = get_available_models(include_online=False)
        
        return (
            "All models restored to original defaults:\n" + "\n".join(success_messages) + "\n\nRestart required for changes to take effect.",
            models['transformer'],
            models['text_encoder'],
            models['text_encoder_2'],
            models['tokenizer'],
            models['tokenizer_2'],
            models['vae'],
            models['feature_extractor'],
            models['image_encoder']
        )
    except Exception as e:
        return f"Error restoring model defaults: {str(e)}", None, None, None, None, None, None, None, None
        

block = gr.Blocks(css=css).queue()
with block:
   
    gr.Markdown('# FramePack (QueueItUp version)')
    with gr.Tabs():
        with gr.Tab("Framepack_QueueItUp"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Gallery(
                        label="Image (adding multiple images will create a jobs for each, leaving image blank will be prompt Text 2 Video",
                        height=500,
                        columns=4,
                        object_fit="contain",
                        elem_classes=["input-gallery"],
                        show_label=True,
                        allow_preview=False,  # Disable built-in preview
                        show_download_button=True,
                        container=True
                    )
                    prompt = gr.Textbox(label="Prompt", value=Config.DEFAULT_PROMPT)
                    n_prompt = gr.Textbox(label="Negative Prompt", value=Config.DEFAULT_N_PROMPT, visible=True)
                    save_prompt_button = gr.Button("Save Prompt to Quick List")

                    quick_list = gr.Dropdown(
                        label="Quick List",
                        choices=[item['prompt'] for item in quick_prompts],
                        value=quick_prompts[0]['prompt'] if quick_prompts else None,
                        allow_custom_value=True
                    )
                    delete_prompt_button = gr.Button("Delete Selected Prompt from Quick List")

                    with gr.Group():
                        use_teacache = gr.Checkbox(label='Use TeaCache', value=Config.DEFAULT_USE_TEACACHE, info='Faster speed, but often makes hands and fingers slightly worse.')
                        seed = gr.Number(label="Seed use -1 to create random seed for job", value=Config.DEFAULT_SEED, precision=0)
                        job_name = gr.Textbox(label="Job Name (optional prefix)", value=Config.DEFAULT_JOB_NAME, info=f"Optional prefix name for this job you can enter source_image_filename or date_time or blank will defaut to Job-")
                        create_job_outputs_folder = gr.Textbox(label="Job Outputs Folder", value=Config.OUTPUTS_FOLDER, info="The path Where the output video for this job will be saved, the default directory is displayed below, optional to change, but a bad path will cause an error.")
                        create_job_history_folder = gr.Textbox(label="Job History Folder", value=Config.JOB_HISTORY_FOLDER, info="The path Where the job history will be saved, the default directory is displayed below, optional to change, but a bad path will cause an error.")
                        video_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=Config.DEFAULT_VIDEO_LENGTH, step=0.1)
                        latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                        steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=Config.DEFAULT_STEPS, step=1, info='Changing this value is not recommended.')
                        cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=Config.DEFAULT_CFG, step=0.01, visible=False)  # Should not change
                        gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=Config.DEFAULT_GS, step=0.01, info='Changing this value is not recommended.')
                        rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=Config.DEFAULT_RS, step=0.01, visible=False)  # Should not change
                        gpu_memory = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=Config.DEFAULT_GPU_MEMORY, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")
                        mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=Config.DEFAULT_MP4_CRF, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")
                        create_job_keep_completed_job = gr.Checkbox(label="Keep this Job when completed", value=Config.KEEP_COMPLETED_JOB, info="If checked, when this job is completed it will stay in the queue as status 'completed' so you can edit them and run them again")
                        keep_temp_png = gr.Checkbox(label="Keep temp PNG file", value=Config.DEFAULT_KEEP_TEMP_PNG, info="If checked, temporary job history PNG file will not be deleted after job is processed")
                        keep_temp_json = gr.Checkbox(label="Keep temp JSON file", value=Config.DEFAULT_KEEP_TEMP_JSON, info="If checked, temporary job history JSON file will not be deleted after job is processed")
                    with gr.Group():
                        save_job_defaults_button = gr.Button(value="Save current job settings as Defaults", interactive=True, elem_id="save_job_defaults_button")
                        restore_job_defaults_button = gr.Button(value="Restore Original job settings", interactive=True, elem_id="restore_job_defaults_button")

                with gr.Column():
                    with gr.Row():
                        queue_button = gr.Button(value="Add to Queue", interactive=True, elem_id="queue_button")
                        start_button = gr.Button(value="Start Queued Jobs", interactive=True, elem_id="start_button")
                        abort_button = gr.Button(value="Abort Generation", interactive=False, elem_id="abort_button")

                    gr.Markdown('Note: video previews will appear here once you click start.')
                    preview_image = gr.Image(label="Latents", visible=False)
                    progress_desc1 = gr.Markdown('', elem_classes=['no-generating-animation', 'progress-desc'], elem_id="progress_desc1")  # Step progress (X/Y steps)
                    progress_bar1 = gr.HTML('', elem_classes=['no-generating-animation', 'progress-bar'], elem_id="progress_bar1")  # Step progress bar
                    result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True, visible=False)
                    progress_desc2 = gr.Markdown('', elem_classes=['no-generating-animation', 'progress-desc'], elem_id="progress_desc2")  # Job progress (X/Y seconds)
                    progress_bar2 = gr.HTML('', elem_classes=['no-generating-animation', 'progress-bar'], elem_id="progress_bar2")  # Job progress bar



                    queue_display = gr.Gallery(
                        label="Job Queue Gallery",
                        show_label=True,
                        columns=5,
                        object_fit="contain",
                        elem_classes=["queue-gallery"],
                        allow_preview=True,
                        container=True
                    )
                    

        with gr.Tab("Edit jobs in the Queue"):
            gr.Markdown("### Queuing Order")
            with gr.Row():
                delete_completed_button = gr.Button(value="Delete Completed Jobs", interactive=True)
                delete_pending_button = gr.Button(value="Delete Pending Jobs", interactive=True)
                delete_failed_button = gr.Button(value="Delete Failed Jobs", interactive=True)
                delete_all_button = gr.Button(value="Delete All Jobs", interactive=True)
            gr.Markdown("Note: Jobs that are Processing are always listed at the top, Jobs that are completed are always at the bottom")
            gr.Markdown("you can edit any job including completed jobs, click the pencil icon make changes and click save, it will switch from completed to pending and can be processed again, if you like the seed check the corresponding job histore json file to grab the seed and change the -1 to the actual seed used")
            # Add edit dialog
            with gr.Row():
                with gr.Column():
                    edit_job_name = gr.Textbox(label="Job ID", visible=False)
                    edit_group = gr.Group(visible=False)  # Hidden by default
                    with edit_group:
                        with gr.Row():
                            with gr.Column(scale=1):
                                save_edit_button = gr.Button("Save Changes")
                            with gr.Column(scale=1):
                                dummy_label = gr.HTML(
                                    '''
                                    <div style="text-align:center; font-weight:bold;">
                                        Editing Job<br>
                                        <span style="font-weight:normal;">Please Click Save or Cancel</span>
                                    </div>
                                    '''
                                )
                            with gr.Column(scale=1):
                                cancel_edit_button = gr.Button("Cancel changes")
                        edit_prompt = gr.Textbox(label="Edit Prompt")
                        edit_n_prompt = gr.Textbox(label="Edit Negative Prompt")
                        edit_video_length = gr.Slider(label="Edit Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                        edit_seed = gr.Number(label="Edit Seed", value=-1, precision=0)
                        edit_use_teacache = gr.Checkbox(label='Edit Use TeaCache', value=True)
                        edit_gpu_memory = gr.Slider(label="Edit GPU Memory Preservation (GB)", minimum=6, maximum=128, value=6, step=0.1)
                        edit_steps = gr.Slider(label="Edit Steps", minimum=1, maximum=100, value=25, step=1)
                        edit_cfg = gr.Slider(label="Edit CFG Scale", visible=False, minimum=1.0, maximum=32.0, value=1.0, step=0.01)
                        edit_gs = gr.Slider(label="Edit Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                        edit_rs = gr.Slider(label="Edit CFG Re-Scale", visible=False, minimum=0.0, maximum=1.0, value=0.0, step=0.01)
                        edit_mp4_crf = gr.Slider(label="Edit MP4 Compression", minimum=0, maximum=100, value=16, step=1)
                        edit_keep_temp_png = gr.Checkbox(label="Edit Keep temp PNG", value=False)
                        edit_keep_temp_json = gr.Checkbox(label="Edit Keep temp JSON", value=False)
                        edit_outputs_folder = gr.Textbox(label="Edit Outputs Folder - (be careful with this setting as it can cause errors if not set correctly)", value=Config.OUTPUTS_FOLDER)
                        edit_job_history_folder = gr.Textbox(label="Edit Job History Folder", value=Config.JOB_HISTORY_FOLDER)
                        edit_keep_temp_mp4 = gr.Checkbox(label="Edit Keep temp MP4", value=Config.KEEP_TEMP_MP4)
                        edit_keep_completed_job = gr.Checkbox(label="Edit Keep Completed Jobs", value=Config.KEEP_COMPLETED_JOB)

            queue_table = gr.DataFrame(
                headers=None,
                datatype=["markdown","str","str","str","str","str","str","str","markdown"],
                col_count=(9, "fixed"),
                value=[],
                interactive=False,
                visible=True,
                elem_classes=["gradio-dataframe"],
                # Optionally, set max_rows or height if available in your Gradio version
            )


        with gr.Tab("Settings"):
            with gr.Tabs():
                with gr.Tab("System settings"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### System Settings")
                            settings_status = gr.Markdown()  # For showing status messages
                            
                            settings_outputs_folder = gr.Textbox(
                                label="Outputs Folder - Default folder where videos are initially saved", 
                                value=Config.OUTPUTS_FOLDER,
                                interactive=False,
                                info="This setting cannot be changed. Videos are initially saved here and then moved to your custom folder if specified in the job settings."
                            )
                            settings_job_history_folder = gr.Textbox(
                                label="Job History Folder (is where the individual job json files and input image are stored with the jobs metadata)", 
                                value=Config.JOB_HISTORY_FOLDER,
                                interactive=False,
                                info="This setting cannot be changed here,  it can be changed on a per job or batch job basis if specified in the job settings or edit settings."
                            )
                            settings_debug_mode = gr.Checkbox(
                                label="Debug Mode", 
                                value=Config.DEBUG_MODE
                            )
                            settings_keep_temp_mp4 = gr.Checkbox(
                                label="Keep temp smaller mp4 leftovers", 
                                value=Config.KEEP_TEMP_MP4
                            )
                            settings_keep_completed_job = gr.Checkbox(
                                label="Keep Completed Jobs", 
                                value=Config.KEEP_COMPLETED_JOB
                            )
                            
                            with gr.Row():
                                save_system_button = gr.Button("Save System Settings", variant="primary")
                                restore_system_button = gr.Button("Restore System Defaults", variant="secondary")
                            
                            # Connect the buttons
                            save_system_button.click(
                                fn=save_system_settings,
                                inputs=[
                                    settings_outputs_folder,
                                    settings_job_history_folder,
                                    settings_debug_mode,
                                    settings_keep_temp_mp4,
                                    settings_keep_completed_job
                                ],
                                outputs=[settings_status]
                            )
                            
                            restore_system_button.click(
                                fn=restore_system_settings,
                                inputs=[],
                                outputs=[
                                    settings_outputs_folder,
                                    settings_job_history_folder,
                                    settings_debug_mode,
                                    settings_keep_temp_mp4,
                                    settings_keep_completed_job,
                                    settings_status
                                ]
                            )

                with gr.Tab("Model Defaults"):
                    with gr.Row():
                        with gr.Column():      
                            gr.Markdown("### Hugging Face Settings")
                            settings_hf_token = gr.Textbox(
                                label="Hugging Face Token", 
                                value=Config.HF_TOKEN,
                                type="text",
                                info="Enter your Hugging Face token to enable downloading models. Get it from https://huggingface.co/settings/tokens"
                            )
                            save_token_button = gr.Button("Save Token")
                            
                            def save_hf_token(token):
                                # First save to settings.ini
                                config = configparser.ConfigParser()
                                config.read(INI_FILE)
                                
                                if 'System Defaults' not in config:
                                    config['System Defaults'] = {}
                                    
                                config['System Defaults']['hf_token'] = token
                                
                                with open(INI_FILE, 'w') as f:
                                    config.write(f)
                                    
                                # Update the global Config instance
                                global Config
                                if Config._instance is None:
                                    Config._instance = Config.from_settings(config)
                                Config._instance.HF_TOKEN = token
                                
                                # Also update the global Config variable
                                Config.HF_TOKEN = token
                                
                                # Reload settings to ensure everything is in sync
                                Config = Config.from_settings(load_settings())
                                
                                return "Token saved successfully!"
                            
                            save_token_button.click(
                                fn=save_hf_token,
                                inputs=[settings_hf_token],
                                outputs=[gr.Markdown()]
                            )
                            
                            gr.Markdown("### Model Selection BETA not really working yet - change at your own risk")
                            with gr.Row():
                                include_online_models = gr.Checkbox(label="Include Online Models", value=False)
                                refresh_models_button = gr.Button("Refresh Model List")
                            
                            available_models = get_available_models(include_online=False)
                            
                            # Create a status display for model operations
                            model_status = gr.Markdown()
                            
                            # Helper function to get display name for a model
                            def get_display_name(model_name):
                                """Convert actual model name to display name if it exists in the available models"""
                                if not model_name:
                                    return None
                                for model_list in available_models.values():
                                    for display_name in model_list:
                                        base_name = (display_name.replace('LOCAL-', '')
                                                              .replace(' - CURRENT DEFAULT MODEL', '')
                                                              .replace(' - ORIGINAL DEFAULT MODEL', '')
                                                              .strip())
                                        if base_name == model_name:
                                            return display_name
                                return model_name
                            
                            # Transformer
                            with gr.Row():
                                with gr.Column(scale=4):
                                    settings_transformer = gr.Dropdown(
                                        label="Transformer",
                                        choices=available_models['transformer'],
                                        value=get_display_name(Config.DEFAULT_TRANSFORMER),
                                        allow_custom_value=True
                                    )
                            
                            # Text Encoder
                            with gr.Row():
                                with gr.Column(scale=4):
                                    settings_text_encoder = gr.Dropdown(
                                        label="Text Encoder",
                                        choices=available_models['text_encoder'],
                                        value=get_display_name(Config.DEFAULT_TEXT_ENCODER),
                                        allow_custom_value=True
                                    )
                            
                            # Text Encoder 2
                            with gr.Row():
                                with gr.Column(scale=4):
                                    settings_text_encoder_2 = gr.Dropdown(
                                        label="Text Encoder 2",
                                        choices=available_models['text_encoder_2'],
                                        value=get_display_name(Config.DEFAULT_TEXT_ENCODER_2),
                                        allow_custom_value=True
                                    )
                            
                            # Tokenizer
                            with gr.Row():
                                with gr.Column(scale=4):
                                    settings_tokenizer = gr.Dropdown(
                                        label="Tokenizer",
                                        choices=available_models['tokenizer'],
                                        value=get_display_name(Config.DEFAULT_TOKENIZER),
                                        allow_custom_value=True
                                    )
                            
                            # Tokenizer 2
                            with gr.Row():
                                with gr.Column(scale=4):
                                    settings_tokenizer_2 = gr.Dropdown(
                                        label="Tokenizer 2",
                                        choices=available_models['tokenizer_2'],
                                        value=get_display_name(Config.DEFAULT_TOKENIZER_2),
                                        allow_custom_value=True
                                    )
                            
                            # VAE
                            with gr.Row():
                                with gr.Column(scale=4):
                                    settings_vae = gr.Dropdown(
                                        label="VAE",
                                        choices=available_models['vae'],
                                        value=get_display_name(Config.DEFAULT_VAE),
                                        allow_custom_value=True
                                    )
                            
                            # Feature Extractor
                            with gr.Row():
                                with gr.Column(scale=4):
                                    settings_feature_extractor = gr.Dropdown(
                                        label="Feature Extractor",
                                        choices=available_models['feature_extractor'],
                                        value=get_display_name(Config.DEFAULT_FEATURE_EXTRACTOR),
                                        allow_custom_value=True
                                    )
                            
                            # Image Encoder
                            with gr.Row():
                                with gr.Column(scale=4):
                                    settings_image_encoder = gr.Dropdown(
                                        label="Image Encoder",
                                        choices=available_models['image_encoder'],
                                        value=get_display_name(Config.DEFAULT_IMAGE_ENCODER),
                                        allow_custom_value=True
                                    )
                            


                            # Add global Set/Restore buttons at the bottom
                            with gr.Row():
                                with gr.Column(scale=1):
                                    set_all_models_button = gr.Button("Set All Models as Default", size="large")
                                with gr.Column(scale=1):
                                    restore_all_models_button = gr.Button("Restore All Model Defaults", size="large")

                            def update_model_lists(include_online):
                                models = get_available_models(include_online=include_online)
                                return [
                                    gr.update(choices=models['transformer']),
                                    gr.update(choices=models['text_encoder']),
                                    gr.update(choices=models['text_encoder_2']),
                                    gr.update(choices=models['tokenizer']),
                                    gr.update(choices=models['tokenizer_2']),
                                    gr.update(choices=models['vae']),
                                    gr.update(choices=models['feature_extractor']),
                                    gr.update(choices=models['image_encoder'])
                                ]

                            # Connect the online model UI elements
                            include_online_models.change(
                                fn=update_model_lists,
                                inputs=[include_online_models],
                                outputs=[
                                    settings_transformer,
                                    settings_text_encoder,
                                    settings_text_encoder_2,
                                    settings_tokenizer,
                                    settings_tokenizer_2,
                                    settings_vae,
                                    settings_feature_extractor,
                                    settings_image_encoder
                                ]
                            )

                            refresh_models_button.click(
                                fn=update_model_lists,
                                inputs=[include_online_models],
                                outputs=[
                                    settings_transformer,
                                    settings_text_encoder,
                                    settings_text_encoder_2,
                                    settings_tokenizer,
                                    settings_tokenizer_2,
                                    settings_vae,
                                    settings_feature_extractor,
                                    settings_image_encoder
                                ]
                            )

                            # Connect the Set All Models button
                            set_all_models_button.click(
                                fn=set_all_models_as_default,
                                inputs=[
                                    settings_transformer,
                                    settings_text_encoder,
                                    settings_text_encoder_2,
                                    settings_tokenizer,
                                    settings_tokenizer_2,
                                    settings_vae,
                                    settings_feature_extractor,
                                    settings_image_encoder
                                ],
                                outputs=[
                                    model_status,
                                    settings_transformer,
                                    settings_text_encoder,
                                    settings_text_encoder_2,
                                    settings_tokenizer,
                                    settings_tokenizer_2,
                                    settings_vae,
                                    settings_feature_extractor,
                                    settings_image_encoder
                                ]
                            )

                            # Connect the Restore All Models button
                            restore_all_models_button.click(
                                fn=restore_all_model_defaults,
                                outputs=[
                                    model_status,
                                    settings_transformer,
                                    settings_text_encoder,
                                    settings_text_encoder_2,
                                    settings_tokenizer,
                                    settings_tokenizer_2,
                                    settings_vae,
                                    settings_feature_extractor,
                                    settings_image_encoder
                                ]
                            )



    save_job_defaults_button.click(
        fn=save_job_defaults_from_ui,
        inputs=[prompt, n_prompt, use_teacache, seed, job_name, video_length, steps, cfg, gs, rs, gpu_memory, mp4_crf, keep_temp_png, keep_temp_json],
        outputs=[gr.Markdown()]
    )

    restore_job_defaults_button.click(
        fn=lambda: restore_job_defaults(),
        inputs=[],
        outputs=[
            prompt, n_prompt, use_teacache, seed, job_name, video_length, steps,
            cfg, gs, rs, gpu_memory, mp4_crf,
            keep_temp_png, keep_temp_json
        ]
    )


    # Connect UI elements
    save_prompt_button.click(
        fn=save_quick_prompt,
        inputs=[prompt, n_prompt, video_length, job_name, gs, steps, use_teacache, seed, cfg, rs, gpu_memory, mp4_crf, keep_temp_png, keep_temp_json],
        outputs=[prompt, quick_list, n_prompt, video_length, job_name, gs, steps, use_teacache, seed, cfg, rs, gpu_memory, mp4_crf, keep_temp_png, keep_temp_json],
        queue=False
    )
    delete_prompt_button.click(
        delete_quick_prompt,
        inputs=[quick_list],
        outputs=[prompt, n_prompt, quick_list, video_length, job_name, gs, steps, use_teacache, seed, cfg, rs, gpu_memory, mp4_crf],
        queue=False
    )
    quick_list.change(
        lambda x, current_n_prompt, current_length, current_job_name, current_gs, current_steps, current_teacache, current_seed, current_cfg, current_rs, current_gpu_memory, current_mp4_crf: (
            x,  # prompt
            next((item.get('n_prompt', current_n_prompt) for item in quick_prompts if item['prompt'] == x), current_n_prompt),  # n_prompt
            next((item.get('length', current_length) for item in quick_prompts if item['prompt'] == x), current_length),  # video_length
            next((item.get('job_name', current_job_name) for item in quick_prompts if item['prompt'] == x), current_job_name),  # job_name
            next((item.get('gs', current_gs) for item in quick_prompts if item['prompt'] == x), current_gs),  # gs
            next((item.get('steps', current_steps) for item in quick_prompts if item['prompt'] == x), current_steps),  # steps
            next((item.get('use_teacache', current_teacache) for item in quick_prompts if item['prompt'] == x), current_teacache),  # use_teacache
            next((item.get('seed', current_seed) for item in quick_prompts if item['prompt'] == x), current_seed),  # seed
            next((item.get('cfg', current_cfg) for item in quick_prompts if item['prompt'] == x), current_cfg),  # cfg
            next((item.get('rs', current_rs) for item in quick_prompts if item['prompt'] == x), current_rs),  # rs
            next((item.get('gpu_memory', current_gpu_memory) for item in quick_prompts if item['prompt'] == x), current_gpu_memory),  # gpu_memory
            next((item.get('mp4_crf', current_mp4_crf) for item in quick_prompts if item['prompt'] == x), current_mp4_crf)  # mp4_crf
        ) if x else (x, current_n_prompt, current_length, current_job_name, current_gs, current_steps, current_teacache, current_seed, current_cfg, current_rs, current_gpu_memory, current_mp4_crf),
        inputs=[quick_list, n_prompt, video_length, job_name, gs, steps, use_teacache, seed, cfg, rs, gpu_memory, mp4_crf],
        outputs=[prompt, n_prompt, video_length, job_name, gs, steps, use_teacache, seed, cfg, rs, gpu_memory, mp4_crf],
        queue=False
    )

    # Load queue on startup
    block.load(
        fn=lambda: (update_queue_table(), update_queue_display()),
        outputs=[queue_table, queue_display]
    )
    # Connect queue actions
    queue_table.select(
        fn=handle_queue_action,
        inputs=[],
        outputs=[
            edit_prompt,
            edit_n_prompt,
            edit_video_length,
            edit_seed,
            edit_use_teacache,
            edit_gpu_memory,
            edit_steps,
            edit_cfg,
            edit_gs,
            edit_rs,
            edit_mp4_crf,
            edit_keep_temp_png,
            edit_keep_temp_json,
            edit_outputs_folder,      # Added
            edit_job_history_folder,  # Added
            edit_keep_temp_mp4,       # Added
            edit_keep_completed_job, # Added
            edit_job_name,
            edit_group
        ]
    ).then(
        fn=lambda: (update_queue_table(), update_queue_display()),
        inputs=[],
        outputs=[queue_table, queue_display]
    )

    delete_completed_button.click(
        fn=delete_completed_jobs,
        inputs=[],
        outputs=[queue_table, queue_display]
    )

    delete_pending_button.click(
        fn=delete_pending_jobs,
        inputs=[],
        outputs=[queue_table, queue_display]
    )
    delete_failed_button.click(
        fn=delete_failed_jobs,
        inputs=[],
        outputs=[queue_table, queue_display]
    )
    delete_all_button.click(
        fn=delete_all_jobs,
        inputs=[],
        outputs=[queue_table, queue_display]
    )
    save_edit_button.click(
        fn=edit_job,
        inputs=[
            edit_job_name, 
            edit_prompt, 
            edit_n_prompt, 
            edit_video_length, 
            edit_seed, 
            edit_use_teacache, 
            edit_gpu_memory, 
            edit_steps, 
            edit_cfg, 
            edit_gs, 
            edit_rs, 
            edit_mp4_crf, 
            edit_keep_temp_png, 
            edit_keep_temp_json,
            edit_outputs_folder,      # Added
            edit_job_history_folder,  # Added
            edit_keep_temp_mp4,       # Added
            edit_keep_completed_job  # Added
        ],
        outputs=[queue_table, queue_display, edit_group]
    )

    cancel_edit_button.click(
        fn=hide_edit_window,
        outputs=[edit_group]
    )

    job_data = [input_image, prompt, n_prompt, seed, video_length, latent_window_size, steps, cfg, gs, rs, gpu_memory, use_teacache, mp4_crf, keep_temp_png, keep_temp_json]
    start_button.click(
        fn=process,
        inputs=[],
        outputs=[
            queue_button, start_button, abort_button,
            preview_image, result_video,
            progress_desc1, progress_bar1,
            progress_desc2, progress_bar2,
            queue_display, queue_table
        ]
    )
    abort_button.click(
        fn=end_process,
        outputs=[queue_table, queue_display, start_button, abort_button, queue_button]
    )


    queue_button.click(
        fn=add_to_queue_handler,
        inputs=[
            input_image, prompt, n_prompt, video_length, seed, job_name, 
            use_teacache, gpu_memory, steps, cfg, gs, rs, mp4_crf, 
            keep_temp_png, keep_temp_json,
            create_job_outputs_folder,
            create_job_history_folder,
            settings_keep_temp_mp4,  # Use the actual Gradio component
            create_job_keep_completed_job
        ],
        outputs=[queue_table, queue_display, queue_button]
    )

# Add these calls at startup
reset_processing_jobs()
cleanup_orphaned_files()


# Launch the interface

block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser
)

# iface = gr.Interface(fn=greet, inputs="text", outputs="text")
# iface.launch(share=True)

# Removed scroll-to-bottom button code

gr.HTML(
    """
    <script>
    function resizeGradioAGGrid() {
        // Find all AG-Grid instances and resize columns to fit
        document.querySelectorAll('.ag-root').forEach(function(grid) {
            if (grid && grid.__agComponent) {
                try {
                    grid.__agComponent.gridOptions.api.sizeColumnsToFit();
                } catch (e) {}
            }
        });
    }
    window.addEventListener('resize', resizeGradioAGGrid);
    // Also trigger once on load
    setTimeout(resizeGradioAGGrid, 1000);
    </script>
    """
)

def create_default_settings():
    """Create default settings.ini file"""
    config = configparser.ConfigParser()
    config['Job Defaults'] = {}
    config['System Defaults'] = {}
    config['Model Defaults'] = {}
    save_settings(config)



block.load(
    fn=lambda: (
        gr.update(interactive=True),      # queue_button
        gr.update(interactive=True),      # start_button
        gr.update(interactive=False),     # abort_button
        gr.update(visible=False),         # preview_image
        gr.update(visible=False),         # result_video
        "",                               # progress_desc1
        "",                               # progress_bar1
        "",                               # progress_desc2
        "",                               # progress_bar2
        update_queue_display(),           # queue_display
        update_queue_table()              # queue_table
    ),
    outputs=[
        queue_button, start_button, abort_button,
        preview_image, result_video,
        progress_desc1, progress_bar1,
        progress_desc2, progress_bar2,
        queue_display, queue_table
    ]
)


# End of file

def restore_model_default(model_type):
    """Restore a specific model to its original default in settings.ini"""
    try:
        # Get original defaults
        original_defaults = Config.get_original_defaults()
        original_value = original_defaults[f'DEFAULT_{model_type.upper()}']
        
        config = configparser.ConfigParser()
        config.read(INI_FILE)
        
        if 'Model Defaults' not in config:
            config['Model Defaults'] = {}
            
        # Update the config
        config['Model Defaults'][f'DEFAULT_{model_type.upper()}'] = repr(original_value)
        
        # Save to file
        with open(INI_FILE, 'w') as f:
            config.write(f)
            
        # Update Config object
        setattr(Config, f'DEFAULT_{model_type.upper()}', original_value)
        
        # Get updated model lists
        models = get_available_models(include_online=False)
        
        # Return status message and individual model lists
        return (
            f"{model_type} restored to original default successfully. Restart required for changes to take effect.",
            models['transformer'],
            models['text_encoder'],
            models['text_encoder_2'],
            models['tokenizer'],
            models['tokenizer_2'],
            models['vae'],
            models['feature_extractor'],
            models['image_encoder']
        )
    except Exception as e:
        return f"Error restoring {model_type} default: {str(e)}", None, None, None, None, None, None, None, None


