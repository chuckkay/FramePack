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
def get_truncated_state_info(state):
    """Helper function to create truncated state info for debugging"""
    preview_info = None
    # if state.last_preview is not None:
        # if isinstance(state.last_preview, np.ndarray):
            # preview_info = f"<Image array with shape={state.last_preview.shape}>"
        # elif isinstance(state.last_preview, Image.Image):
            # preview_info = f"<PIL Image with size={state.last_preview.size}>"
        # else:
            # preview_info = "<Unknown image format>"
            
    return {
        'is_processing': state.is_processing,
        'current_job_name': state.current_job_name,
        # 'last_preview': preview_info,
        'last_progress': state.last_progress,
        'last_progress_html': '...' if state.last_progress_html else None
    }

def get_available_models():
    """Get list of available models from hub directory"""
    hub_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_download", "hub")
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
    
    if os.path.exists(hub_dir):
        for item in os.listdir(hub_dir):
            if os.path.isdir(os.path.join(hub_dir, item)):
                # Map models to their correct categories
                if "hunyuanvideo" in item.lower():
                    models['text_encoder'].append(item)
                    models['text_encoder_2'].append(item)
                    models['tokenizer'].append(item)
                    models['tokenizer_2'].append(item)
                    models['vae'].append(item)
                elif "flux_redux" in item.lower():
                    models['feature_extractor'].append(item)
                    models['image_encoder'].append(item)
                elif "framepack" in item.lower():
                    models['transformer'].append(item)
    
    return models

# Path to settings file
SETTINGS_FILE = os.path.join(os.getcwd(), 'settings.ini')

# Path to the quick prompts JSON file
PROMPT_FILE = os.path.join(os.getcwd(), 'quick_prompts.json')

# Queue file path
QUEUE_FILE = os.path.join(os.getcwd(), 'job_queue.json')

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
# Global variables
debug_mode = False
keep_completed_jobs = True

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
    with open(SETTINGS_FILE, 'w') as f:
        config.write(f)

def save_ips_defaults_from_ui(use_teacache, seed, video_length, steps, cfg, gs, rs, gpu_memory, mp4_crf,
                           keep_temp_png, keep_temp_mp4, keep_temp_json,
                           model_name, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                           vae, feature_extractor, image_encoder, transformer):
    """Save IPS defaults from UI settings"""
    config = load_settings()
    if 'IPS Defaults' not in config:
        config['IPS Defaults'] = {}
    section = config['IPS Defaults']
    
    # Save IPS defaults
    section['DEFAULT_USE_TEACACHE'] = repr(use_teacache)
    section['DEFAULT_SEED'] = repr(seed)
    section['DEFAULT_VIDEO_LENGTH'] = repr(video_length)
    section['DEFAULT_STEPS'] = repr(steps)
    section['DEFAULT_CFG'] = repr(cfg)
    section['DEFAULT_GS'] = repr(gs)
    section['DEFAULT_RS'] = repr(rs)
    section['DEFAULT_GPU_MEMORY'] = repr(gpu_memory)
    section['DEFAULT_MP4_CRF'] = repr(mp4_crf)
    section['DEFAULT_KEEP_TEMP_PNG'] = repr(keep_temp_png)
    section['DEFAULT_KEEP_TEMP_MP4'] = repr(keep_temp_mp4)
    section['DEFAULT_KEEP_TEMP_JSON'] = repr(keep_temp_json)
    
    # Save model settings
    if 'Model Settings' not in config:
        config['Model Settings'] = {}
    section = config['Model Settings']
    section['DEFAULT_MODEL_NAME'] = repr(model_name)
    section['DEFAULT_TEXT_ENCODER'] = repr(text_encoder)
    section['DEFAULT_TEXT_ENCODER_2'] = repr(text_encoder_2)
    section['DEFAULT_TOKENIZER'] = repr(tokenizer)
    section['DEFAULT_TOKENIZER_2'] = repr(tokenizer_2)
    section['DEFAULT_VAE'] = repr(vae)
    section['DEFAULT_FEATURE_EXTRACTOR'] = repr(feature_extractor)
    section['DEFAULT_IMAGE_ENCODER'] = repr(image_encoder)
    section['DEFAULT_TRANSFORMER'] = repr(transformer)
    
    save_settings(config)
    return "Settings saved successfully!"

@dataclass
class Config:
    """Centralized configuration for default values"""
    # Default prompt settings
    DEFAULT_PROMPT: str = None
    DEFAULT_NEGATIVE_PROMPT: str = None
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
    DEFAULT_KEEP_TEMP_MP4: bool = None
    DEFAULT_KEEP_TEMP_JSON: bool = None

    # Model settings
    DEFAULT_MODEL_NAME: str = None
    DEFAULT_TEXT_ENCODER: str = None
    DEFAULT_TEXT_ENCODER_2: str = None
    DEFAULT_TOKENIZER: str = None
    DEFAULT_TOKENIZER_2: str = None
    DEFAULT_VAE: str = None
    DEFAULT_FEATURE_EXTRACTOR: str = None
    DEFAULT_IMAGE_ENCODER: str = None
    DEFAULT_TRANSFORMER: str = None

    # System defaults
    OUTPUTS_FOLDER: str = None
    JOB_HISTORY_FOLDER: str = None
    DEBUG_MODE: bool = None
    KEEP_COMPLETED_JOBS: bool = None

    @classmethod
    def get_original_defaults(cls):
        """Returns a dictionary of original default values - this is the single source of truth for defaults"""
        return {
            'DEFAULT_PROMPT': "The girl dances gracefully, with clear movements, full of charm.",
            'DEFAULT_NEGATIVE_PROMPT': "",
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
            'DEFAULT_KEEP_TEMP_MP4': False,
            'DEFAULT_KEEP_TEMP_JSON': True,
            'DEFAULT_MODEL_NAME': "models--lllyasviel--FramePackI2V_HY",
            'DEFAULT_TEXT_ENCODER': "models--hunyuanvideo-community--HunyuanVideo",
            'DEFAULT_TEXT_ENCODER_2': "models--hunyuanvideo-community--HunyuanVideo",
            'DEFAULT_TOKENIZER': "models--hunyuanvideo-community--HunyuanVideo",
            'DEFAULT_TOKENIZER_2': "models--hunyuanvideo-community--HunyuanVideo",
            'DEFAULT_VAE': "models--hunyuanvideo-community--HunyuanVideo",
            'DEFAULT_FEATURE_EXTRACTOR': "models--lllyasviel--flux_redux_bfl",
            'DEFAULT_IMAGE_ENCODER': "models--lllyasviel--flux_redux_bfl",
            'DEFAULT_TRANSFORMER': "models--lllyasviel--FramePackI2V_HY",
            'OUTPUTS_FOLDER': './outputs/',
            'JOB_HISTORY_FOLDER': './job_history/',
            'DEBUG_MODE': False,
            'KEEP_COMPLETED_JOBS': True
        }

    @classmethod
    def from_settings(cls, config):
        """Create Config instance from settings.ini values"""
        # Load IPS Defaults section
        section = config['IPS Defaults']
        defaults = cls.get_original_defaults()
        
        # Load all values from settings, using defaults as fallback
        for key, default_value in defaults.items():
            if key.startswith('DEFAULT_'):
                try:
                    value = section.get(key, repr(default_value))
                    # Only use ast.literal_eval for non-string values
                    if isinstance(default_value, (bool, int, float)):
                        setattr(cls, key, ast.literal_eval(value))
                    else:
                        setattr(cls, key, value.strip("'"))
                except (KeyError, ValueError):
                    setattr(cls, key, default_value)
                    section[key] = repr(default_value)
                    save_settings(config)

        # Load Model Settings section
        if 'Model Settings' not in config:
            config['Model Settings'] = {}
        section = config['Model Settings']
        model_defaults = [
            'DEFAULT_MODEL_NAME', 'DEFAULT_TEXT_ENCODER', 'DEFAULT_TEXT_ENCODER_2',
            'DEFAULT_TOKENIZER', 'DEFAULT_TOKENIZER_2', 'DEFAULT_VAE',
            'DEFAULT_FEATURE_EXTRACTOR', 'DEFAULT_IMAGE_ENCODER', 'DEFAULT_TRANSFORMER'
        ]
        for key in model_defaults:
            try:
                value = section.get(key, repr(defaults[key]))
                setattr(cls, key, value.strip("'"))
            except (KeyError, ValueError):
                setattr(cls, key, defaults[key])
                section[key] = repr(defaults[key])
                save_settings(config)

        # Load System Defaults section
        if 'System Defaults' not in config:
            config['System Defaults'] = {}
        section = config['System Defaults']
        
        # Load system values from settings, using defaults as fallback
        for key, default_value in defaults.items():
            if not key.startswith('DEFAULT_'):
                try:
                    value = section.get(key, str(default_value))
                    # Only use ast.literal_eval for non-string values
                    if isinstance(default_value, (bool, int, float)):
                        setattr(cls, key, ast.literal_eval(value))
                    else:
                        setattr(cls, key, value.strip("'"))
                except (KeyError, ValueError):
                    setattr(cls, key, default_value)
                    section[key] = str(default_value)
                    save_settings(config)

        return cls

    @classmethod
    def to_settings(cls, config):
        """Save Config instance values to settings.ini"""
        # Save IPS Defaults section
        section = config['IPS Defaults']
        ips_defaults = [
            'DEFAULT_PROMPT', 'DEFAULT_NEGATIVE_PROMPT', 'DEFAULT_VIDEO_LENGTH',
            'DEFAULT_GS', 'DEFAULT_STEPS', 'DEFAULT_USE_TEACACHE', 'DEFAULT_SEED',
            'DEFAULT_CFG', 'DEFAULT_RS', 'DEFAULT_GPU_MEMORY', 'DEFAULT_MP4_CRF',
            'DEFAULT_KEEP_TEMP_PNG', 'DEFAULT_KEEP_TEMP_MP4', 'DEFAULT_KEEP_TEMP_JSON'
        ]
        for key in ips_defaults:
            section[key] = repr(getattr(cls, key))

        # Save Model Settings section
        if 'Model Settings' not in config:
            config['Model Settings'] = {}
        section = config['Model Settings']
        model_defaults = [
            'DEFAULT_MODEL_NAME', 'DEFAULT_TEXT_ENCODER', 'DEFAULT_TEXT_ENCODER_2',
            'DEFAULT_TOKENIZER', 'DEFAULT_TOKENIZER_2', 'DEFAULT_VAE',
            'DEFAULT_FEATURE_EXTRACTOR', 'DEFAULT_IMAGE_ENCODER', 'DEFAULT_TRANSFORMER'
        ]
        for key in model_defaults:
            section[key] = repr(getattr(cls, key))

        # Save System Defaults section
        if 'System Defaults' not in config:
            config['System Defaults'] = {}
        section = config['System Defaults']
        system_defaults = ['OUTPUTS_FOLDER', 'JOB_HISTORY_FOLDER', 'DEBUG_MODE', 'KEEP_COMPLETED_JOBS']
        for key in system_defaults:
            section[key] = str(getattr(cls, key))

        save_settings(config)

    @classmethod
    def get_default_prompt_tuple(cls):
        """Returns a tuple of all default values in the correct order"""
        return (
            cls.DEFAULT_PROMPT,
            cls.DEFAULT_NEGATIVE_PROMPT,
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
            cls.DEFAULT_KEEP_TEMP_MP4,
            cls.DEFAULT_KEEP_TEMP_JSON
        )

    @classmethod
    def get_default_prompt_dict(cls):
        """Returns a dictionary of default values for quick prompts"""
        return {
            'prompt': cls.DEFAULT_PROMPT,
            'n_prompt': cls.DEFAULT_NEGATIVE_PROMPT,
            'length': cls.DEFAULT_VIDEO_LENGTH,
            'gs': cls.DEFAULT_GS,
            'steps': cls.DEFAULT_STEPS,
            'use_teacache': cls.DEFAULT_USE_TEACACHE,
            'seed': cls.DEFAULT_SEED,
            'cfg': cls.DEFAULT_CFG,
            'rs': cls.DEFAULT_RS,
            'gpu_memory_preservation': cls.DEFAULT_GPU_MEMORY,
            'mp4_crf': cls.DEFAULT_MP4_CRF,
            'keep_temp_png': cls.DEFAULT_KEEP_TEMP_PNG,
            'keep_temp_mp4': cls.DEFAULT_KEEP_TEMP_MP4,
            'keep_temp_json': cls.DEFAULT_KEEP_TEMP_JSON
        }

def load_settings():
    """Load settings from settings.ini file and ensure all sections and values exist"""
    config = configparser.ConfigParser()
    
    # Get default values
    default_values = Config.get_original_defaults()
    
    # Create default sections if file doesn't exist
    if not os.path.exists(SETTINGS_FILE):
        config['IPS Defaults'] = {k: repr(v) for k, v in default_values.items() if k.startswith('DEFAULT_')}
        config['System Defaults'] = {k: str(v) for k, v in default_values.items() if not k.startswith('DEFAULT_')}
        with open(SETTINGS_FILE, 'w') as f:
            config.write(f)
    else:
        # Read existing config
        config.read(SETTINGS_FILE)
        
        # Ensure IPS Defaults section exists with all values
        if 'IPS Defaults' not in config:
            config['IPS Defaults'] = {}
        
        # Check and add any missing values in IPS Defaults
        for key, value in default_values.items():
            if key.startswith('DEFAULT_') and key not in config['IPS Defaults']:
                config['IPS Defaults'][key] = repr(value)
        
        # Ensure System Defaults section exists with all values
        if 'System Defaults' not in config:
            config['System Defaults'] = {}
        
        # Check and add any missing system values
        for key, value in default_values.items():
            if not key.startswith('DEFAULT_') and key not in config['System Defaults']:
                config['System Defaults'][key] = str(value)
        
        # Save any changes made to the config
        with open(SETTINGS_FILE, 'w') as f:
            config.write(f)
    
    return config

def save_settings_from_ui(outputs_folder, job_history_folder, debug_mode, keep_completed_jobs,
                         model_name, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                         vae, feature_extractor, image_encoder, transformer):
    """Save settings from UI inputs"""
    settings_config = configparser.ConfigParser()
    
    # Create all required sections
    settings_config['IPS Defaults'] = {}
    settings_config['Model Settings'] = {}
    settings_config['System Defaults'] = {}
    
    # System Settings
    settings_config['System Defaults'] = {
        'OUTPUTS_FOLDER': repr(outputs_folder),
        'JOB_HISTORY_FOLDER': repr(job_history_folder),
        'DEBUG_MODE': repr(debug_mode),
        'KEEP_COMPLETED_JOBS': repr(keep_completed_jobs)
    }
    
    # Model Settings
    settings_config['Model Settings'] = {
        'DEFAULT_MODEL_NAME': repr(model_name),
        'DEFAULT_TEXT_ENCODER': repr(text_encoder),
        'DEFAULT_TEXT_ENCODER_2': repr(text_encoder_2),
        'DEFAULT_TOKENIZER': repr(tokenizer),
        'DEFAULT_TOKENIZER_2': repr(tokenizer_2),
        'DEFAULT_VAE': repr(vae),
        'DEFAULT_FEATURE_EXTRACTOR': repr(feature_extractor),
        'DEFAULT_IMAGE_ENCODER': repr(image_encoder),
        'DEFAULT_TRANSFORMER': repr(transformer)
    }
    
    # Update global Config object
    Config.OUTPUTS_FOLDER = outputs_folder
    Config.JOB_HISTORY_FOLDER = job_history_folder
    Config.DEBUG_MODE = debug_mode
    Config.KEEP_COMPLETED_JOBS = keep_completed_jobs
    Config.DEFAULT_MODEL_NAME = model_name
    Config.DEFAULT_TEXT_ENCODER = text_encoder
    Config.DEFAULT_TEXT_ENCODER_2 = text_encoder_2
    Config.DEFAULT_TOKENIZER = tokenizer
    Config.DEFAULT_TOKENIZER_2 = tokenizer_2
    Config.DEFAULT_VAE = vae
    Config.DEFAULT_FEATURE_EXTRACTOR = feature_extractor
    Config.DEFAULT_IMAGE_ENCODER = image_encoder
    Config.DEFAULT_TRANSFORMER = transformer
    
    Config.to_settings(settings_config)
    return "Settings saved successfully. Restart required for changes to take effect."

def restore_original_defaults(return_ips_defaults=False):
    """Restore original default values"""
    defaults = Config.get_original_defaults()
    if return_ips_defaults:
        return [
            defaults['DEFAULT_USE_TEACACHE'],
            defaults['DEFAULT_SEED'],
            defaults['DEFAULT_VIDEO_LENGTH'],
            defaults['DEFAULT_STEPS'],
            defaults['DEFAULT_CFG'],
            defaults['DEFAULT_GS'],
            defaults['DEFAULT_RS'],
            defaults['DEFAULT_GPU_MEMORY'],
            defaults['DEFAULT_MP4_CRF'],
            defaults['DEFAULT_KEEP_TEMP_PNG'],
            defaults['DEFAULT_KEEP_TEMP_MP4'],
            defaults['DEFAULT_KEEP_TEMP_JSON'],
            defaults['DEFAULT_MODEL_NAME'],
            defaults['DEFAULT_TEXT_ENCODER'],
            defaults['DEFAULT_TEXT_ENCODER_2'],
            defaults['DEFAULT_TOKENIZER'],
            defaults['DEFAULT_TOKENIZER_2'],
            defaults['DEFAULT_VAE'],
            defaults['DEFAULT_FEATURE_EXTRACTOR'],
            defaults['DEFAULT_IMAGE_ENCODER'],
            defaults['DEFAULT_TRANSFORMER']
        ]
    else:
        return [
            defaults['OUTPUTS_FOLDER'],
            defaults['JOB_HISTORY_FOLDER'],
            defaults['DEBUG_MODE'],
            defaults['KEEP_COMPLETED_JOBS'],
            defaults['DEFAULT_MODEL_NAME'],
            defaults['DEFAULT_TEXT_ENCODER'],
            defaults['DEFAULT_TEXT_ENCODER_2'],
            defaults['DEFAULT_TOKENIZER'],
            defaults['DEFAULT_TOKENIZER_2'],
            defaults['DEFAULT_VAE'],
            defaults['DEFAULT_FEATURE_EXTRACTOR'],
            defaults['DEFAULT_IMAGE_ENCODER'],
            defaults['DEFAULT_TRANSFORMER']
        ]

def save_system_settings_from_ui(outputs_folder, job_history_folder, debug_mode, keep_completed_jobs):
    """Save system settings from UI inputs"""
    global Config, settings_config
    
    # Update config
    config = configparser.ConfigParser()
    config.read(settings_config)
    config['System']['outputs_folder'] = outputs_folder
    config['System']['job_history_folder'] = job_history_folder
    config['System']['debug_mode'] = str(debug_mode)
    config['System']['keep_completed_jobs'] = str(keep_completed_jobs)
    
    with open(settings_config, 'w') as f:
        config.write(f)
    
    # Update Config object
    Config.OUTPUTS_FOLDER = outputs_folder
    Config.JOB_HISTORY_FOLDER = job_history_folder
    Config.DEBUG_MODE = debug_mode
    Config.KEEP_COMPLETED_JOBS = keep_completed_jobs
    
    return f"Settings saved successfully!\nOutputs Folder: {outputs_folder}\nJob History Folder: {job_history_folder}\nDebug Mode: {debug_mode}\nKeep Completed Jobs: {keep_completed_jobs}"

def restore_system_defaults(outputs_folder, job_history_folder, debug_mode, keep_completed_jobs):
    """Restore system settings to original defaults"""
    global Config, settings_config
    
    # Get default values
    defaults = Config.get_original_defaults()
    outputs_folder = defaults['OUTPUTS_FOLDER']
    job_history_folder = defaults['JOB_HISTORY_FOLDER']
    
    # Create folders if they don't exist
    os.makedirs(outputs_folder, exist_ok=True)
    os.makedirs(job_history_folder, exist_ok=True)
    
    Config.OUTPUTS_FOLDER = outputs_folder
    Config.JOB_HISTORY_FOLDER = job_history_folder
    Config.DEBUG_MODE = False
    Config.KEEP_COMPLETED_JOBS = True
    Config.to_settings(settings_config)
    
    # Update local variables
    setup_local_variables()
    
    return "System settings have been restored to defaults."

def save_queue():
    """Save queue state to JSON file"""
    try:
        jobs = [job.to_dict() for job in job_queue]
        with open(QUEUE_FILE, 'w') as f:
            json.dump(jobs, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving queue: {str(e)}")
        traceback.print_exc()
        return False

def load_queue():
    """Load queue state from JSON file"""
    try:
        if os.path.exists(QUEUE_FILE):
            with open(QUEUE_FILE, 'r') as f:
                jobs = json.load(f)
            
            # Clear existing queue and load jobs from file
            job_queue.clear()
            for job_data in jobs:
                job = QueuedJob.from_dict(job_data)
                if job is not None:
                    job_queue.append(job)
            
            return job_queue
        else:
            debug_print("No queue file found")
        return []
    except Exception as e:
        alert_print(f"Error loading queue: {str(e)}")
        traceback.print_exc()
        return []

def setup_local_variables():
    """Set up local variables from Config values"""
    global job_history_folder, outputs_folder, debug_mode, keep_completed_jobs
    job_history_folder = Config.JOB_HISTORY_FOLDER
    outputs_folder = Config.OUTPUTS_FOLDER
    debug_mode = Config.DEBUG_MODE
    keep_completed_jobs = Config.KEEP_COMPLETED_JOBS

# Initialize settings
settings_config = load_settings()
Config = Config.from_settings(settings_config)

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
        'length': Config.DEFAULT_VIDEO_LENGTH,
        'gs': Config.DEFAULT_GS,
        'steps': Config.DEFAULT_STEPS,
        'use_teacache': Config.DEFAULT_USE_TEACACHE,
        'seed': Config.DEFAULT_SEED,
        'cfg': Config.DEFAULT_CFG,
        'rs': Config.DEFAULT_RS,
        'gpu_memory_preservation': Config.DEFAULT_GPU_MEMORY,
        'mp4_crf': Config.DEFAULT_MP4_CRF,
        'keep_temp_png': Config.DEFAULT_KEEP_TEMP_PNG,
        'keep_temp_mp4': Config.DEFAULT_KEEP_TEMP_MP4,
        'keep_temp_json': Config.DEFAULT_KEEP_TEMP_JSON
    }
]

# Load existing prompts or create the file with defaults
if os.path.exists(PROMPT_FILE):
    with open(PROMPT_FILE, 'r') as f:
        quick_prompts = json.load(f)
else:
    quick_prompts = DEFAULT_PROMPTS.copy()
    with open(PROMPT_FILE, 'w') as f:
        json.dump(quick_prompts, f, indent=2)

@dataclass
class QueuedJob:
    prompt: str
    image_path: str
    video_length: float
    job_name: str 
    seed: int
    use_teacache: bool
    gpu_memory_preservation: float
    steps: int
    cfg: float
    gs: float
    rs: float
    n_prompt: str
    status: str = "pending"
    thumbnail: str = ""
    mp4_crf: float = 16
    keep_temp_png: bool = False
    keep_temp_mp4: bool = False
    keep_temp_json: bool = False

    def to_dict(self):
        try:
            return {
                'prompt': self.prompt,
                'image_path': self.image_path,
                'video_length': self.video_length,
                'job_name': self.job_name,
                'seed': self.seed,
                'use_teacache': self.use_teacache,
                'gpu_memory_preservation': self.gpu_memory_preservation,
                'steps': self.steps,
                'cfg': self.cfg,
                'gs': self.gs,
                'rs': self.rs,
                'n_prompt': self.n_prompt, 
                'status': self.status,
                'thumbnail': self.thumbnail,
                'mp4_crf': self.mp4_crf,
                'keep_temp_png': self.keep_temp_png,
                'keep_temp_mp4': self.keep_temp_mp4,
                'keep_temp_json': self.keep_temp_json
            }
        except Exception as e:
            alert_print(f"Error converting job to dict: {str(e)}")
            return None

    @classmethod
    def from_dict(cls, data):
        try:
            return cls(
                prompt=data['prompt'],
                image_path=data['image_path'],
                video_length=data['video_length'],
                job_name=data['job_name'],
                seed=data['seed'],
                use_teacache=data['use_teacache'],
                gpu_memory_preservation=data['gpu_memory_preservation'],
                steps=data['steps'],
                cfg=data['cfg'],
                gs=data['gs'],
                rs=data['rs'],
                n_prompt=data['n_prompt'],
                status=data['status'],
                thumbnail=data['thumbnail'],
                mp4_crf=data['mp4_crf'],
                keep_temp_png=data.get('keep_temp_png', False),  # Default to False for backward compatibility
                keep_temp_mp4=data.get('keep_temp_mp4', False),   # Default to False for backward compatibility
                keep_temp_json=data.get('keep_temp_json', False)   # Default to False for backward compatibility
            )
        except Exception as e:
            alert_print(f"Error creating job from dict: {str(e)}")
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

def add_to_queue(prompt, n_prompt, input_image, video_length, seed, use_teacache, gpu_memory_preservation, steps, cfg, gs, rs, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json, job_name, status="pending"):
    """Add a new job to the queue"""
    try:
               
        hex_id = uuid.uuid4().hex[:8]
        job_name = f"{job_name}_{hex_id}"
        load_queue()

        # Handle text-to-video case
        if input_image is None:
            job = QueuedJob(
                prompt=prompt,
                image_path="text2video",  # Set to None for text-to-video
                video_length=video_length,
                job_name=job_name,
                seed=seed,
                use_teacache=use_teacache,
                gpu_memory_preservation=gpu_memory_preservation,
                steps=steps,
                cfg=cfg,
                gs=gs,
                rs=rs,
                n_prompt=n_prompt,
                status=status,
                mp4_crf=mp4_crf,
                keep_temp_png=keep_temp_png,
                keep_temp_mp4=keep_temp_mp4,
                keep_temp_json=keep_temp_json
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
                gpu_memory_preservation=gpu_memory_preservation,
                steps=steps,
                cfg=cfg,
                gs=gs,
                rs=rs,
                n_prompt=n_prompt,
                status=status,
                mp4_crf=mp4_crf,
                keep_temp_png=keep_temp_png,
                keep_temp_mp4=keep_temp_mp4,
                keep_temp_json=keep_temp_json
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
        
        # Remove completed jobs if keep_completed_jobs is False
        if not keep_completed_jobs:
            completed_jobs_count = len([job for job in job_queue if job.status == "completed"])
            job_queue = [job for job in job_queue if job.status != "completed"]
            # Only print if jobs were actually removed
            if completed_jobs_count > 0:
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
                quick_prompts[0]['length'],
                quick_prompts[0].get('gs', Config.DEFAULT_GS),
                quick_prompts[0].get('steps', Config.DEFAULT_STEPS),
                quick_prompts[0].get('use_teacache', Config.DEFAULT_USE_TEACACHE),
                quick_prompts[0].get('seed', Config.DEFAULT_SEED),
                quick_prompts[0].get('cfg', Config.DEFAULT_CFG),
                quick_prompts[0].get('rs', Config.DEFAULT_RS),
                quick_prompts[0].get('gpu_memory_preservation', Config.DEFAULT_GPU_MEMORY),
                quick_prompts[0].get('mp4_crf', Config.DEFAULT_MP4_CRF),
                quick_prompts[0].get('keep_temp_png', Config.DEFAULT_KEEP_TEMP_PNG),
                quick_prompts[0].get('keep_temp_mp4', Config.DEFAULT_KEEP_TEMP_MP4),
                quick_prompts[0].get('keep_temp_json', Config.DEFAULT_KEEP_TEMP_JSON)
            )
        return Config.get_default_prompt_tuple()
    except Exception as e:
        alert_print(f"Error getting default prompt: {str(e)}")
        return Config.get_default_prompt_tuple()

def save_quick_prompt(prompt_text, n_prompt_text, video_length, gs_value, steps_value, use_teacache_value, seed_value, cfg_value, rs_value, gpu_memory_preservation_value, mp4_crf_value, keep_temp_png_value, keep_temp_mp4_value, keep_temp_json_value):
    global quick_prompts
    if prompt_text:
        # Check if prompt already exists
        for item in quick_prompts:
            if item['prompt'] == prompt_text:
                item['n_prompt'] = n_prompt_text
                item['length'] = video_length
                item['gs'] = gs_value
                item['steps'] = steps_value
                item['use_teacache'] = use_teacache_value
                item['seed'] = seed_value
                item['cfg'] = cfg_value
                item['rs'] = rs_value
                item['gpu_memory_preservation'] = gpu_memory_preservation_value
                item['mp4_crf'] = mp4_crf_value
                item['keep_temp_png'] = keep_temp_png_value
                item['keep_temp_mp4'] = keep_temp_mp4_value
                item['keep_temp_json'] = keep_temp_json_value
                break
        else:
            quick_prompts.append({
                'prompt': prompt_text,
                'n_prompt': n_prompt_text,
                'length': video_length,
                'gs': gs_value,
                'steps': steps_value,
                'use_teacache': use_teacache_value,
                'seed': seed_value,
                'cfg': cfg_value,
                'rs': rs_value,
                'gpu_memory_preservation': gpu_memory_preservation_value,
                'mp4_crf': mp4_crf_value,
                'keep_temp_png': keep_temp_png_value,
                'keep_temp_mp4': keep_temp_mp4_value,
                'keep_temp_json': keep_temp_json_value
            })
        
        with open(PROMPT_FILE, 'w') as f:
            json.dump(quick_prompts, f, indent=2)
    # Keep the text in the prompt box and set it as selected in quick list
    return prompt_text, n_prompt_text, gr.update(choices=[item['prompt'] for item in quick_prompts], value=prompt_text), video_length, gs_value, steps_value, use_teacache_value, seed_value, cfg_value, rs_value, gpu_memory_preservation_value, mp4_crf_value, keep_temp_png_value, keep_temp_mp4_value, keep_temp_json_value

def delete_quick_prompt(prompt_text):
    global quick_prompts
    if prompt_text:
        quick_prompts = [item for item in quick_prompts if item['prompt'] != prompt_text]
        with open(PROMPT_FILE, 'w') as f:
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
    if job.keep_temp_mp4:
        debug_print(f"Keeping temporary MP4 files for job {job_name} as requested")
    if job.keep_temp_json:
        debug_print(f"Keeping temporary JSON file for job {job_name} as requested")

    # Delete the PNG file
    png_path = os.path.join(job_history_folder, f'{job_name}.png')
    try:
        if os.path.exists(png_path) and not job.keep_temp_png:
            os.remove(png_path)
            debug_print(f"Deleted PNG file: {png_path}")
    except OSError as e:
        alert_print(f"Failed to delete PNG file {png_path}: {e}")

    # Delete the job_name.JSON job file
    json_path = os.path.join(job_history_folder, f'{job_name}.json')
    try:
        if os.path.exists(json_path) and not job.keep_temp_json:
            os.remove(json_path)
            debug_print(f"Deleted JSON file: {json_path}")
    except OSError as e:
        alert_print(f"Failed to delete JSON file {json_path}: {e}")

    # regex to grab the trailing number
    pattern = re.compile(rf'^{re.escape(job_name)}_(\d+)\.mp4$')
    candidates = []
    #
    # scan directory
    for fname in os.listdir(outputs_folder):
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
        if count != highest_count and not (job.keep_temp_mp4 and fname.endswith('.mp4')):
            path = os.path.join(outputs_folder, fname)
            try:
                os.remove(path)
            except OSError as e:
                alert_print(f"Failed to delete {fname}: {e}")

    # Rename the remaining MP4 to {job_name}.mp4
    if highest_fname:
        old_path = os.path.join(outputs_folder, highest_fname)
        new_path = os.path.join(outputs_folder, f"{job_name}.mp4")
        try:
            if os.path.exists(new_path):
                os.remove(new_path)  # Remove existing file if it exists
            os.rename(old_path, new_path)
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
        mp4_path = os.path.join(outputs_folder, f"{completed_job.job_name}.mp4")
        extract_thumb_from_processing_mp4(completed_job, mp4_path)
        # img = Image.open(completed_job.thumbnail)
        # width, height = img.size
        # new_height = 200
        # new_width = int((new_height / height) * width)
        # img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # # Create a new image with padding
        # new_img = Image.new('RGB', (200, 200), color='black')
        # new_img.paste(img, ((200 - img.width) // 2, (200 - img.height) // 2))
        # # Add status text if provided
        # status_overlay = "DONE"
        # draw = ImageDraw.Draw(new_img)
        # draw.text((100, 100), status_overlay, fill='yellow', anchor="mm", font=font)           
            # # Save thumbnail
        # thumbnail_path = os.path.join(temp_queue_images, f"thumb_{completed_job.job_name}.png")
        # new_img.save(thumbnail_path)
        # debug_print(f"thumbnail saved {thumbnail_path}")
        # completed_job.thumbnail = thumbnail_path
        # save_queue()
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
def worker(input_image, prompt, n_prompt, seed, job_name, video_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json, next_job):
    #Worker function to process a job
    global stream
    #stream = initialize_stream()
    debug_print(f"Starting worker for job {job_name}")
    debug_print(f"Worker - Initial parameters: video_length={video_length}, steps={steps}, seed={seed}")
    debug_print(f"Worker - Using stream object: {id(stream)}")

    total_latent_sections = (video_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    # job_failed = None not used yet
    # job_id = generate_timestamp() #not used yet
    debug_print(f"Worker - Total latent sections to process: {total_latent_sections}")

    stream.output_queue.push(('progress', (None, "Initializing...", make_progress_bar_html(0, "Step Progress"), "Starting job...", make_progress_bar_html(0, "Job Progress"))))
    debug_print("Worker - Initial progress update pushed")
    debug_print("Worker - Progress update pushed to queue")

    # Save the input image with metadata
    metadata = PngInfo()
    metadata.add_text("prompt", prompt)
    metadata.add_text("n_prompt", n_prompt) 
    metadata.add_text("seed", str(seed))  # This will now be the random seed if it was -1
    metadata.add_text("video_length", str(video_length))
    metadata.add_text("latent_window_size", str(latent_window_size))
    metadata.add_text("steps", str(steps))
    metadata.add_text("cfg", str(cfg))
    metadata.add_text("gs", str(gs))
    metadata.add_text("rs", str(rs))
    metadata.add_text("gpu_memory_preservation", str(gpu_memory_preservation))
    metadata.add_text("use_teacache", str(use_teacache))
    metadata.add_text("mp4_crf", str(mp4_crf))

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

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image
        stream.output_queue.push(('progress', (None, "Image processing...", make_progress_bar_html(0, "Step Progress"), "Processing...", make_progress_bar_html(0, "Job Progress"))))

        # Handle text-to-video case
        if input_image is None:
            # Create a blank image for text-to-video with default resolution
            default_resolution = 640  # Default resolution for text-to-video
            input_image_np = np.zeros((default_resolution, default_resolution, 3), dtype=np.uint8)
            height = width = default_resolution
        else:
            # Handle image-to-video case
            input_image_np = np.array(input_image)
            H, W, C = input_image_np.shape
            height, width = find_nearest_bucket(H, W, resolution=640)
            input_image_np = resize_and_center_crop(input_image_np, target_width=width, target_height=height)

            Image.fromarray(input_image_np).save(os.path.join(job_history_folder, f'{job_name}.png'), pnginfo=metadata)

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

        rnd = torch.Generator("cpu").manual_seed(seed)

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
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
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
                step_percentage = int(100.0 * current_step / steps)
                step_desc = f'Step {current_step} of {steps}'
                step_progress = make_progress_bar_html(step_percentage, f'Step Progress: {step_percentage}%')

                current_time = max(0, (total_generated_latent_frames * 4 - 3) / 30)
                job_percentage = int((current_time / video_length) * 100)
                job_type = "Image to Video" if next_job and next_job.image_path != "text2video" else "Text 2 Video"
                job_desc = f'Creating a {job_type} for job name {job_name} , with these values seed: {seed} cfg scale:{gs} teacache:{use_teacache} mp4_crf:{mp4_crf} \\n Created {current_time:.1f} second(s) of the {video_length} second video - ({job_percentage}% complete)'
                job_progress = make_progress_bar_html(job_percentage, f'Job Progress: {job_percentage}%')

                stream.output_queue.push(('progress', (preview, step_desc, step_progress, job_desc, job_progress)))
                return
            indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=latent_window_size * 4 - 3,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
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
                section_latent_frames = latent_window_size * 2
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_name}_{total_generated_latent_frames}.mp4')
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
            
            stream.output_queue.push(('file', output_filename))

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

    except:
        traceback.print_exc()
        
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    completed_job = next_job
    debug_print(f"Worker - Pushing done state for job: {completed_job.job_name if completed_job else 'None'}")
    stream.output_queue.push(('done', completed_job))

def extract_thumb_from_processing_mp4(next_job, output_filename):
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

            # Overlay centered status text
            text = (f"{status_overlay}")
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1
            thickness = 2

            # Calculate text size to center it
            (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
            x = (thumb.shape[1] - text_w) // 2
            y = (thumb.shape[0] + text_h) // 2

            # Add a black outline to make text more readable
            cv2.putText(
                thumb,
                text,
                (x, y),
                font,
                scale,
                (0, 0, 0),  # Black outline
                thickness + 2,
                cv2.LINE_AA
            )

            # Add the colored text
            cv2.putText(
                thumb,
                text,
                (x, y),
                font,
                scale,
                status_color,
                thickness,
                cv2.LINE_AA
            )

            thumb_path = os.path.join(temp_queue_images, f'thumb_{next_job.job_name}.png')
            cv2.imwrite(thumb_path, thumb)
        cap.release()
    return



def process(input_image, prompt, n_prompt, seed, video_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json, process_state):
    global stream
    stream = initialize_stream()
    
    # Load process state
    state = ProcessState.from_json(process_state)
    debug_print(f"Process - Loaded state: is_processing={state.is_processing}, current_job_name={state.current_job_name}")

    # If we're already processing and have a current job, restore the UI state
    if state.is_processing and state.current_job_name:
        debug_print(f"Process - Restoring state for job: {state.current_job_name}")
        yield (
            gr.update(interactive=True),      # queue_button
            gr.update(interactive=False),     # start_button
            gr.update(interactive=True),      # abort_button
            gr.update(visible=True, value=state.last_preview),  # preview_image
            gr.update(visible=True, value=state.current_video),  # result_video
            state.last_progress or f"processing job {state.current_job_name}...",  # progress_desc
            state.last_progress_html or "",   # progress_bar
            update_queue_display(),           # queue_display
            update_queue_table(),             # queue_table
            state.to_json()                   # process_state
        )
        return

    # Set processing state to True when starting
    state.is_processing = True
    save_queue()  # Save the processing state
    debug_print(f"Process - Using stream object: {id(stream)}")

    # First check for pending jobs
    pending_jobs = [job for job in job_queue if job.status.lower() == "pending"]

    if not pending_jobs:
        # No pending jobs
        state.is_processing = False
        yield (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),
            None,
            None,
            "no pending jobs to process",
            '',
            update_queue_display(),
            update_queue_table(),
            state.to_json()
        )
        return

    # Process first pending job
    next_job = pending_jobs[0]
    queue_table_update, queue_display_update = mark_job_processing(next_job)
    save_queue()
    job_name = next_job.job_name
    
    # Update process state with current job
    state.current_job_name = job_name

    # Handle text-to-video case
    if next_job.image_path == "text2video":
        process_image = None
    else:
        try:
            process_image = np.array(Image.open(next_job.image_path))
        except Exception as e:
            alert_print(f"ERROR loading image: {str(e)}")
            traceback.print_exc()
            raise

    # Use job parameters with defaults if missing
    process_prompt = next_job.prompt if hasattr(next_job, 'prompt') else prompt
    process_n_prompt = next_job.n_prompt if hasattr(next_job, 'n_prompt') else n_prompt
    process_seed = next_job.seed if hasattr(next_job, 'seed') else seed
    debug_print(f"Job {next_job.job_name} initial seed value: {process_seed}")
    
    # Generate random seed if seed is -1
    if process_seed == -1:
        seed = random.randint(0, 2**32 - 1)
        debug_print(f"Generated new random seed for job {next_job.job_name}: {seed}")
        save_queue()
    else:
        seed = process_seed

    # Get remaining job parameters
    process_job_name = next_job.job_name
    process_length = next_job.video_length if hasattr(next_job, 'video_length') else video_length
    process_steps = next_job.steps if hasattr(next_job, 'steps') else steps
    process_cfg = next_job.cfg if hasattr(next_job, 'cfg') else cfg
    process_gs = next_job.gs if hasattr(next_job, 'gs') else gs
    process_rs = next_job.rs if hasattr(next_job, 'rs') else rs
    process_gpu_memory_preservation = next_job.gpu_memory_preservation if hasattr(next_job, 'gpu_memory_preservation') else gpu_memory_preservation
    process_teacache = next_job.use_teacache if hasattr(next_job, 'use_teacache') else use_teacache
    process_keep_temp_png = next_job.keep_temp_png if hasattr(next_job, 'keep_temp_png') else keep_temp_png
    process_keep_temp_mp4 = next_job.keep_temp_mp4 if hasattr(next_job, 'keep_temp_mp4') else keep_temp_mp4
    process_keep_temp_json = next_job.keep_temp_json if hasattr(next_job, 'keep_temp_json') else keep_temp_json
    process_mp4_crf = next_job.mp4_crf if hasattr(next_job, 'mp4_crf') else mp4_crf

    # Save job parameters to JSON if enabled
    if process_keep_temp_json:
        job_params = {
            'prompt': process_prompt,
            'negative_prompt': process_n_prompt,
            'seed': seed,
            'job_name': process_job_name,
            'length': process_length,
            'steps': process_steps,
            'cfg': process_cfg,
            'gs': process_gs,
            'rs': process_rs,
            'gpu_memory_preservation': process_gpu_memory_preservation,
            'use_teacache': process_teacache,
            'mp4_crf': process_mp4_crf,
        }
        json_path = os.path.join(job_history_folder, f'{process_job_name}.json')
        with open(json_path, 'w') as f:
            json.dump(job_params, f, indent=2)

    # Start processing
    debug_print(f"Starting worker for job {next_job.job_name}")
    async_run(worker, process_image, process_prompt, process_n_prompt, seed, process_job_name, 
             process_length, latent_window_size, process_steps, 
             process_cfg, process_gs, process_rs, 
             process_gpu_memory_preservation, process_teacache, process_mp4_crf,
             process_keep_temp_png, process_keep_temp_mp4, process_keep_temp_json, next_job)

    # Initial yield - Modified to ensure UI elements are visible and properly initialized
    yield (
        gr.update(interactive=True),      # queue_button
        gr.update(interactive=False),     # start_button
        gr.update(interactive=True),      # abort_button
        gr.update(visible=True),          # preview_image (Start: visible)
        gr.update(value=None),            # result_video
        "Initializing steps...",          # progress_desc1 (step progress)
        make_progress_bar_html(0, "Preparing"),  # progress_bar1 (step progress)
        "Starting job processing...",     # progress_desc2 (job progress)
        make_progress_bar_html(0, "Job Progress"),  # progress_bar2 (job progress)
        update_queue_display(),           # queue_display
        update_queue_table(),             # queue_table
        state.to_json()                   # process_state
    )

    # Process output queue
    while True:
        try:
            flag, data = stream.output_queue.next()
            #debug_print(f"Process - After stream.output_queue.next(), got flag: {flag}")

            if flag == 'file':
                debug_print("[DEBUG] Process - Handling file flag")
                output_filename = data
                
                # Ensure path is absolute
                if not os.path.isabs(output_filename):
                    output_filename = os.path.abspath(output_filename)
                
                # Store video path in state
                state.current_video = output_filename
                
                extract_thumb_from_processing_mp4(next_job, output_filename)
                
                # Update progress state for video segment
                state.last_step_description = "Processing video segment..."
                state.last_step_progress = make_progress_bar_html(100, "Video segment complete")
                state.last_job_description = "Processing video segment..."
                state.last_job_progress = make_progress_bar_html(100, "Video segment complete")
                
                yield (
                    gr.update(interactive=True),    # queue_button
                    gr.update(interactive=False),   # start_button
                    gr.update(interactive=True),    # abort_button
                    gr.update(visible=True),        # preview_image (File Output: visible)
                    gr.update(value=state.current_video),  # result_video
                    state.last_step_description,    # keep last step progress
                    state.last_step_progress,       # keep last step progress bar
                    state.last_job_description,     # keep last job progress
                    state.last_job_progress,        # keep last job progress bar
                    update_queue_display(),         # queue_display
                    update_queue_table(),           # queue_table
                    state.to_json()                 # process_state
                )

            if flag == 'progress':
#push to progress - None, Start sampling...,  progress_bar Step Progress, Processing..., make_progress_bar_html Job Progress
                preview, step_desc, step_progress, job_desc, job_progress = data
                # try:
                    # # Convert preview to PIL Image if it's a numpy array
                    # preview_update = Image.fromarray(preview) if isinstance(preview, np.ndarray) else preview
                    # if preview_update:
                        # # Save preview to temp file
                        #WHYYYYYYYYYYYYYYYYYYYYYYYYY
                        # preview_path = os.path.join(temp_queue_images, f"preview_{state.current_job_name}.png")
                        # preview_update.save(preview_path)
                        # preview_update = preview_path  # Use the file path instead of PIL Image
                # except Exception as e:
                    # debug_print(f"Error handling preview: {str(e)}")
                    # preview_update = None

                # Update state before yielding
                # state.last_preview = preview_update
                state.last_step_description = step_desc
                state.last_step_progress = step_progress
                state.last_job_description = job_desc
                state.last_job_progress = job_progress
                
                yield (
                    gr.update(interactive=True),    # queue_button
                    gr.update(interactive=False),   # start_button
                    gr.update(interactive=True),    # abort_button
                    gr.update(visible=True, value=preview), # preview_image (Progress: visible)
                    gr.update(),                    # leave result_video as is
                    step_desc,                      # progress_desc1 (step progress)
                    step_progress,                  # progress_bar1 (step progress)
                    job_desc,                       # progress_desc2 (job progress)
                    job_progress,                   # progress_bar2 (job progress)
                    update_queue_display(),         # queue_display
                    update_queue_table(),           # queue_table
                    state.to_json()                 # process_state
                )


            if flag == 'abort':
                if stream.input_queue.top() == 'abort':
                    aborted_job = next((job for job in job_queue if job.status == "processing"), None)
                    if aborted_job:
                        clean_up_temp_mp4png(aborted_job)
                        mp4_path = os.path.join(outputs_folder, f"{aborted_job.job_name}.mp4") 
                        extract_thumb_from_processing_mp4(aborted_job, mp4_path)
                        queue_table_update, queue_display_update = mark_job_pending(aborted_job)
                        save_queue()
                        
                        state.is_processing = False
                        state.current_job_name = None
                        # state.last_preview = None
                        state.last_progress = "Job Aborted"
                        state.last_progress_html = None
                        
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
                            update_queue_table(),           # queue_table
                            state.to_json()                 # process_state
                        )

                return


            if flag == 'done':
                completed_job = data
                print(f"completed job recieved at done flag job name {completed_job.job_name}")


                # previous job completed
                state.is_processing = True
                state.current_job_name = completed_job.job_name
                # state.last_preview = None
                state.last_progress = "Job Complete"
                state.last_progress_html = make_progress_bar_html(100, "Complete")
                clean_up_temp_mp4png(completed_job)
                mp4_path = os.path.join(outputs_folder, f"{completed_job.job_name}.mp4")
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
                    # Handle NULL image path (text-to-video)
                    if next_job.image_path == "text2video":
                        next_image = None
                    else:
                        next_image = np.array(Image.open(next_job.image_path))
                        
                    # Use job parameters with defaults if missing
                    next_prompt = next_job.prompt if hasattr(next_job, 'prompt') else prompt
                    next_n_prompt = next_job.n_prompt if hasattr(next_job, 'n_prompt') else n_prompt
                    next_seed = next_job.seed if hasattr(next_job, 'seed') else seed
                    if next_seed == -1:
                        seed = random.randint(0, 2**32 - 1)
                        debug_print(f"Generated new random seed for job {next_job.job_name}: {seed}")
                        save_queue()

                    next_job_name = next_job.job_name
                    next_length = next_job.video_length if hasattr(next_job, 'video_length') else video_length
                    next_steps = next_job.steps if hasattr(next_job, 'steps') else steps
                    next_cfg = next_job.cfg if hasattr(next_job, 'cfg') else cfg
                    next_gs = next_job.gs if hasattr(next_job, 'gs') else gs
                    next_rs = next_job.rs if hasattr(next_job, 'rs') else rs
                    next_gpu_memory_preservation = next_job.gpu_memory_preservation if hasattr(next_job, 'gpu_memory_preservation') else gpu_memory_preservation
                    next_teacache = next_job.use_teacache if hasattr(next_job, 'use_teacache') else use_teacache
                    next_keep_temp_png = next_job.keep_temp_png if hasattr(next_job, 'keep_temp_png') else keep_temp_png
                    next_keep_temp_mp4 = next_job.keep_temp_mp4 if hasattr(next_job, 'keep_temp_mp4') else keep_temp_mp4
                    next_keep_temp_json = next_job.keep_temp_json if hasattr(next_job, 'keep_temp_json') else keep_temp_json
                    next_mp4_crf = next_job.mp4_crf if hasattr(next_job, 'mp4_crf') else mp4_crf
                    yield (
                        gr.update(interactive=True),    # queue_button
                        gr.update(interactive=False),    # start_button
                        gr.update(interactive=True),   # abort_button
                        gr.update(visible=True),       # preview_image 
                        gr.update(value=state.current_video),  # show result_video with final file
                        "Generation Complete",          # progress_desc1 (step progress)
                        make_progress_bar_html(100, "Complete"),  # progress_bar1 (step progress)
                        "Job Finished Successfully",    # progress_desc2 (job progress)
                        make_progress_bar_html(100, "Complete"),  # progress_bar2 (job progress)
                        update_queue_display(),         # queue_display
                        update_queue_table(),           # queue_table
                        state.to_json()                 # process_state
                    )
                        

                    # Save job parameters to JSON if enabled
                    if next_keep_temp_json:
                        job_params = {
                            'prompt': next_prompt,
                            'negative_prompt': next_n_prompt,
                            'seed': seed,
                            'job_name': next_job_name,
                            'length': next_length,
                            'steps': next_steps,
                            'cfg': next_cfg,
                            'gs': next_gs,
                            'rs': next_rs,
                            'gpu_memory_preservation': next_gpu_memory_preservation,
                            'use_teacache': next_teacache,
                            'mp4_crf': next_mp4_crf,
                        }
                        json_path = os.path.join(job_history_folder, f'{next_job_name}.json')
                        with open(json_path, 'w') as f:
                            json.dump(job_params, f, indent=2)
                    debug_print(f"Starting worker for job {next_job.job_name}")
                    async_run(worker, next_image, next_prompt, next_n_prompt, seed, next_job_name,
                            next_length, latent_window_size, next_steps,
                            next_cfg, next_gs, next_rs,
                            next_gpu_memory_preservation, next_teacache, next_mp4_crf,
                            next_keep_temp_png, next_keep_temp_mp4, next_keep_temp_json, next_job)
                    # Immediately yield initial UI state for the new job
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
                        update_queue_table(),             # queue_table
                        state.to_json()                   # process_state
                    )

                else:
                    debug_print("No more pending jobs to process")
                    yield (
                        gr.update(interactive=True),   # queue_button (always enabled)
                        gr.update(interactive=True),   # start_button
                        gr.update(interactive=False),  # abort_button
                        None,  # preview_image
                        gr.update(value=state.current_video),  # show result_video with final file
                        "No more pending jobs to process",  # progress_desc
                        "Job Finished Successfully",    # progress_desc2 (job progress)
                        update_queue_display(),        # queue_display
                        update_queue_table(),         # queue_table
                        state.to_json()                # process_state
                    )
                    return

        except Exception as e:
            debug_print(f"Error in process loop: {str(e)}")
            state.is_processing = False
            state.current_job_name = None

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
    

def add_to_queue_handler(input_image, prompt, n_prompt, video_length, seed, job_name, use_teacache, gpu_memory_preservation, steps, cfg, gs, rs, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json):
    """Handle adding a new job to the queue"""
    try:
        if prompt is None and input_image is None:
            return (
                None,  # result_video
                None,  # preview_image
                "No prompt and no input image provided",  # progress_desc
                '',    # progress_bar
                gr.update(interactive=True),   # start_button
                gr.update(interactive=False),  # abort_button
                gr.update(interactive=True),   # queue_button (always enabled)
                update_queue_table(),         # queue_table
                update_queue_display()        # queue_display
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
                gpu_memory_preservation=gpu_memory_preservation,
                steps=steps,
                cfg=cfg,
                gs=gs,
                rs=rs,
                status="pending",
                mp4_crf=mp4_crf,
                keep_temp_png=keep_temp_png,
                keep_temp_mp4=keep_temp_mp4,
                keep_temp_json=keep_temp_json
            )
            save_queue()
            return update_queue_table(), update_queue_display(), gr.update(interactive=True)

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
                    gpu_memory_preservation=gpu_memory_preservation,
                    steps=steps,
                    cfg=cfg,
                    gs=gs,
                    rs=rs,
                    status="pending",
                    mp4_crf=mp4_crf,
                    keep_temp_png=keep_temp_png,
                    keep_temp_mp4=keep_temp_mp4,
                    keep_temp_json=keep_temp_json
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
                gpu_memory_preservation=gpu_memory_preservation,
                steps=steps,
                cfg=cfg,
                gs=gs,
                rs=rs,
                status="pending",
                mp4_crf=mp4_crf,
                keep_temp_png=keep_temp_png,
                keep_temp_mp4=keep_temp_mp4,
                keep_temp_json=keep_temp_json
            )
        
            job = next((job for job in job_queue if job.job_name == job_name), None)
            if job and job.image_path:
                job.thumbnail = create_thumbnail(job, status_change=True)
                save_queue()  # Save after changing statuses

        return update_queue_table(), update_queue_display(), gr.update(interactive=True)  # queue_button (always enabled)
    except Exception as e:
        alert_print(f"Error in add_to_queue_handler: {str(e)}")
        traceback.print_exc()
        return update_queue_table(), update_queue_display(), gr.update(interactive=True)  # queue_button (always enabled)

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
    if evt.index is None or evt.value not in ["⏫️", "⬆️", "⬇️", "⏬️", "❌", "📋", "✎"]:
        return (
            Config.DEFAULT_PROMPT,
            Config.DEFAULT_NEGATIVE_PROMPT,
            Config.DEFAULT_VIDEO_LENGTH,
            Config.DEFAULT_SEED,
            Config.DEFAULT_USE_TEACACHE,
            Config.DEFAULT_GPU_MEMORY,
            Config.DEFAULT_STEPS,
            Config.DEFAULT_CFG,
            Config.DEFAULT_GS,
            Config.DEFAULT_RS,
            Config.DEFAULT_MP4_CRF,
            Config.DEFAULT_KEEP_TEMP_PNG,
            Config.DEFAULT_KEEP_TEMP_MP4,
            Config.DEFAULT_KEEP_TEMP_JSON,
            "",  # job_name
            gr.update(visible=False)  # edit group visibility
        )
    
    row_index, col_index = evt.index
    button_clicked = evt.value
    
    # Get the job ID from the first column
    job_name = job_queue[row_index].job_name
    
    if button_clicked == "⏫️":  # Double up arrow (Top)
        move_job_to_top(job_name)
    elif button_clicked == "⬆️":  # Single up arrow (Up)
        move_job(job_name, 'up')
    elif button_clicked == "⬇️":  # Single down arrow (Down)
        move_job(job_name, 'down')
    elif button_clicked == "⏬️":  # Double down arrow (Bottom)
        move_job_to_bottom(job_name)
    elif button_clicked == "❌":
        remove_job(job_name)
    elif button_clicked == "📋":
        copy_job(job_name)
    elif button_clicked == "✎":
        # Get the job
        job = next((j for j in job_queue if j.job_name == job_name), None)
        if job and job.status in ["pending", "completed"]:  # Allow editing both pending and completed jobs
            # Return the job parameters for editing and show edit group
            return (
            job.prompt,
            job.n_prompt,
            job.video_length,
            job.seed,
            job.use_teacache,
            job.gpu_memory_preservation,
            job.steps,
            job.cfg,
            job.gs,
            job.rs,
            job.mp4_crf,
            job.keep_temp_png,
            job.keep_temp_mp4,
                job.keep_temp_json,
                job_name,
                gr.update(visible=True)  # Show edit group
            )
    
    # For all other actions, return default values
    return (
        Config.DEFAULT_PROMPT,
        Config.DEFAULT_NEGATIVE_PROMPT,
        Config.DEFAULT_VIDEO_LENGTH,
        Config.DEFAULT_SEED,
        Config.DEFAULT_USE_TEACACHE,
        Config.DEFAULT_GPU_MEMORY,
        Config.DEFAULT_STEPS,
        Config.DEFAULT_CFG,
        Config.DEFAULT_GS,
        Config.DEFAULT_RS,
        Config.DEFAULT_MP4_CRF,
        Config.DEFAULT_KEEP_TEMP_PNG,
        Config.DEFAULT_KEEP_TEMP_MP4,
        Config.DEFAULT_KEEP_TEMP_JSON,
        "",  # job_name
        gr.update(visible=False)  # edit group visibility
    )

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
            gpu_memory_preservation=original_job.gpu_memory_preservation,
            steps=original_job.steps,
            cfg=original_job.cfg,
            gs=original_job.gs,
            rs=original_job.rs,
            n_prompt=original_job.n_prompt,
            status="pending",
            thumbnail="",
            mp4_crf=original_job.mp4_crf,
            keep_temp_png=original_job.keep_temp_png,
            keep_temp_mp4=original_job.keep_temp_mp4,
            keep_temp_json=original_job.keep_temp_json
        )
        
        # Find the original job's index
        original_index = job_queue.index(original_job)
        
        # Insert the new job right after the original
        job_queue.insert(original_index + 1, new_job)
        
        # Create thumbnail for the new job
        if new_image_path:
            new_job.thumbnail = create_thumbnail(new_job)
        
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


def edit_job(job_name, new_prompt, new_n_prompt, new_video_length, new_seed, new_use_teacache, new_gpu_memory_preservation, new_steps, new_cfg, new_gs, new_rs, new_mp4_crf, new_keep_temp_png, new_keep_temp_mp4, new_keep_temp_json):
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
                job.gpu_memory_preservation = new_gpu_memory_preservation
                job.steps = new_steps
                job.cfg = new_cfg
                job.gs = new_gs
                job.rs = new_rs
                job.mp4_crf = new_mp4_crf
                job.keep_temp_png = new_keep_temp_png
                job.keep_temp_mp4 = new_keep_temp_mp4
                job.keep_temp_json = new_keep_temp_json
                
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

class ProcessState:
    def __init__(self):
        self.is_processing = False
        self.current_job_name = None
        # self.last_preview = None
        self.last_step_description = None
        self.last_step_progress = None
        self.last_job_description = None
        self.last_job_progress = None
        self.current_video = None
        
    def to_json(self):
        return {
            "is_processing": self.is_processing,
            "current_job_name": self.current_job_name,
            # "last_preview": self.last_preview,
            "last_step_description": self.last_step_description,
            "last_step_progress": self.last_step_progress,
            "last_job_description": self.last_job_description,
            "last_job_progress": self.last_job_progress,
            "current_video": self.current_video
        }

    @classmethod
    def from_json(cls, data):
        state = cls()
        if data:
            state.is_processing = data.get("is_processing", False)
            state.current_job_name = data.get("current_job_name")
            # state.last_preview = data.get("last_preview")
            state.last_step_description = data.get("last_step_description")
            state.last_step_progress = data.get("last_step_progress")
            state.last_job_description = data.get("last_job_description")
            state.last_job_progress = data.get("last_job_progress")
            state.current_video = data.get("current_video")
        return state

block = gr.Blocks(css=css).queue()

with block:
    # Add browser state for process tracking
    process_state = gr.BrowserState(
        default_value=ProcessState().to_json(),
        storage_key="framepack_process_state"
    )
    
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
                        show_download_button=False,
                        container=True
                    )
                    prompt = gr.Textbox(label="Prompt", value='')
                    n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=True)
                    save_prompt_button = gr.Button("Save Prompt to Quick List")
                    quick_list = gr.Dropdown(
                        label="Quick List",
                        choices=[item['prompt'] for item in quick_prompts],
                        value=quick_prompts[0]['prompt'] if quick_prompts else None,
                        allow_custom_value=True
                    )
                    delete_prompt_button = gr.Button("Delete Selected Prompt from Quick List")


                    with gr.Group():
                        use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                        seed = gr.Number(label="Seed use -1 to create random seed for job", value=-1, precision=0)
                        job_name = gr.Textbox(label="Job Name (optional prefix)", value="", info="Optional prefix name for this job")
                        video_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                        latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                        steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                        cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                        gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                        rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                        gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                        mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")
                        keep_temp_png = gr.Checkbox(label="Keep temp PNG file", value=False, info="If checked, temporary job history PNG file will not be deleted after job is processed")
                        keep_temp_mp4 = gr.Checkbox(label="Keep temp MP4 files", value=False, info="If checked, extra temp MP4 files will not be deleted after job is processed")
                        keep_temp_json = gr.Checkbox(label="Keep temp JSON file", value=False, info="If checked, temporary job histor JSON file will not be deleted after job is processed")

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

                    # Add state change event handlers
                    def update_ui_from_state(state_json):
                        state = ProcessState.from_json(state_json)
                        if state.is_processing:
                            # Convert preview if it's a numpy array
                            # preview_value = Image.fromarray(state.last_preview) if isinstance(state.last_preview, np.ndarray) else state.last_preview
                            return {
                                queue_button: gr.update(interactive=True),
                                start_button: gr.update(interactive=False),
                                abort_button: gr.update(interactive=True),
                                preview_image: gr.update(visible=True),##### big test removal#, value=preview),#value=preview_value),
                                result_video: gr.update(visible=True, value=state.current_video) if state.current_video else gr.update(visible=False),
                                progress_desc1: state.last_step_description or "Initializing...",
                                progress_bar1: state.last_step_progress or make_progress_bar_html(0, "Step Progress"),
                                progress_desc2: state.last_job_description or "Starting job...",
                                progress_bar2: state.last_job_progress or make_progress_bar_html(0, "Job Progress")
                            }
                        return {
                            queue_button: gr.update(interactive=True),
                            start_button: gr.update(interactive=True),
                            abort_button: gr.update(interactive=False),
                            preview_image: gr.update(visible=False),
                            result_video: gr.update(visible=False),
                            progress_desc1: "",
                            progress_bar1: "",
                            progress_desc2: "",
                            progress_bar2: ""
                        }
                    
                    # Connect state changes to UI updates
                    process_state.change(
                        fn=update_ui_from_state,
                        inputs=[process_state],
                        outputs=[queue_button, start_button, abort_button, preview_image, result_video, progress_desc1, progress_bar1, progress_desc2, progress_bar2]
                    )

                    queue_display = gr.Gallery(
                        label="Job Queue Gallery",
                        show_label=True,
                        columns=5,
                        object_fit="contain",
                        elem_classes=["queue-gallery"],
                        allow_preview=True,
                        show_download_button=False,
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
                        edit_gpu_memory_preservation = gr.Slider(label="Edit GPU Memory Preservation (GB)", minimum=6, maximum=128, value=6, step=0.1)
                        edit_steps = gr.Slider(label="Edit Steps", minimum=1, maximum=100, value=25, step=1)
                        edit_cfg = gr.Slider(label="Edit CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01)
                        edit_gs = gr.Slider(label="Edit Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                        edit_rs = gr.Slider(label="Edit CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01)
                        edit_mp4_crf = gr.Slider(label="Edit MP4 Compression", minimum=0, maximum=100, value=16, step=1)
                        edit_keep_temp_png = gr.Checkbox(label="Edit Keep temp PNG", value=False)
                        edit_keep_temp_mp4 = gr.Checkbox(label="Edit Keep temp MP4", value=False)
                        edit_keep_temp_json = gr.Checkbox(label="Edit Keep temp JSON", value=False)
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
                with gr.Tab("Job Defaults"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### this will set the new job defaults, REQUIRES RESTART, these changes will not change setting for jobs that are already in the queue")
                            with gr.Row():
                                save_defaults_button = gr.Button("Save job settings as Defaults")
                                restore_defaults_button = gr.Button("Restore Original job settings")
                            settings_use_teacache = gr.Checkbox(label='Use TeaCache', value=Config.DEFAULT_USE_TEACACHE)
                            settings_seed = gr.Number(label="Seed", value=Config.DEFAULT_SEED, precision=0)
                            settings_video_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=Config.DEFAULT_VIDEO_LENGTH, step=0.1)
                            settings_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=Config.DEFAULT_STEPS, step=1)
                            settings_cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=Config.DEFAULT_CFG, step=0.01)
                            settings_gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=Config.DEFAULT_GS, step=0.01)
                            settings_rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=Config.DEFAULT_RS, step=0.01)
                            settings_gpu_memory = gr.Slider(label="GPU Memory Preservation (GB)", minimum=6, maximum=128, value=Config.DEFAULT_GPU_MEMORY, step=0.1)
                            settings_mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=Config.DEFAULT_MP4_CRF, step=1)
                            settings_keep_temp_png = gr.Checkbox(label="Keep temp PNG", value=Config.DEFAULT_KEEP_TEMP_PNG)
                            settings_keep_temp_mp4 = gr.Checkbox(label="Keep temp MP4", value=Config.DEFAULT_KEEP_TEMP_MP4)
                            settings_keep_temp_json = gr.Checkbox(label="Keep temp JSON", value=Config.DEFAULT_KEEP_TEMP_JSON)


                with gr.Tab("Global System Defaults"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### this will set the new system defaults, it REQUIRES RESTART to take")
                            with gr.Row():
                                save_settings_button = gr.Button("Save Settings")
                                restore_defaults_button = gr.Button("Restore Defaults")
                            settings_status = gr.Markdown()
                            
                            settings_outputs_folder = gr.Textbox(label="Outputs Folder", value=Config.OUTPUTS_FOLDER)
                            settings_job_history_folder = gr.Textbox(label="Job History Folder this is where the job settings json file and job input image is stored", value=Config.JOB_HISTORY_FOLDER)
                            settings_debug_mode = gr.Checkbox(label="Debug Mode", value=Config.DEBUG_MODE)
                            settings_keep_completed_jobs = gr.Checkbox(label="Keep Completed Jobs", value=Config.KEEP_COMPLETED_JOBS)
                            
                            gr.Markdown("### Model Selection")
                            available_models = get_available_models()
                            settings_model_name = gr.Dropdown(
                                label="Default Model",
                                choices=available_models['transformer'],
                                value=Config.DEFAULT_MODEL_NAME
                            )
                            settings_text_encoder = gr.Dropdown(
                                label="Text Encoder",
                                choices=available_models['text_encoder'],
                                value=Config.DEFAULT_TEXT_ENCODER
                            )
                            settings_text_encoder_2 = gr.Dropdown(
                                label="Text Encoder 2",
                                choices=available_models['text_encoder_2'],
                                value=Config.DEFAULT_TEXT_ENCODER_2
                            )
                            settings_tokenizer = gr.Dropdown(
                                label="Tokenizer",
                                choices=available_models['tokenizer'],
                                value=Config.DEFAULT_TOKENIZER
                            )
                            settings_tokenizer_2 = gr.Dropdown(
                                label="Tokenizer 2",
                                choices=available_models['tokenizer_2'],
                                value=Config.DEFAULT_TOKENIZER_2
                            )
                            settings_vae = gr.Dropdown(
                                label="VAE",
                                choices=available_models['vae'],
                                value=Config.DEFAULT_VAE
                            )
                            settings_feature_extractor = gr.Dropdown(
                                label="Feature Extractor",
                                choices=available_models['feature_extractor'],
                                value=Config.DEFAULT_FEATURE_EXTRACTOR
                            )
                            settings_image_encoder = gr.Dropdown(
                                label="Image Encoder",
                                choices=available_models['image_encoder'],
                                value=Config.DEFAULT_IMAGE_ENCODER
                            )
                            settings_transformer = gr.Dropdown(
                                label="Transformer",
                                choices=available_models['transformer'],
                                value=Config.DEFAULT_TRANSFORMER
                            )
                            
                            # Settings event handlers
                            save_settings_button.click(
                                fn=save_settings_from_ui,
                                inputs=[
                                    settings_outputs_folder,
                                    settings_job_history_folder,
                                    settings_debug_mode,
                                    settings_keep_completed_jobs,
                                    settings_model_name,
                                    settings_text_encoder,
                                    settings_text_encoder_2,
                                    settings_tokenizer,
                                    settings_tokenizer_2,
                                    settings_vae,
                                    settings_feature_extractor,
                                    settings_image_encoder,
                                    settings_transformer
                                ],
                                outputs=[settings_status]
                            )
                            
                            restore_defaults_button.click(
                                fn=lambda: restore_original_defaults(return_ips_defaults=False),
                                inputs=[],
                                outputs=[
                                    settings_outputs_folder,
                                    settings_job_history_folder,
                                    settings_debug_mode,
                                    settings_keep_completed_jobs,
                                    settings_model_name,
                                    settings_text_encoder,
                                    settings_text_encoder_2,
                                    settings_tokenizer,
                                    settings_tokenizer_2,
                                    settings_vae,
                                    settings_feature_extractor,
                                    settings_image_encoder,
                                    settings_transformer
                                ]
                            )

    # Connect settings buttons and all other UI event bindings at the top level (not in a nested with block)
    save_defaults_button.click(
        fn=save_ips_defaults_from_ui,
        inputs=[
            settings_use_teacache, settings_seed, settings_video_length, settings_steps,
            settings_cfg, settings_gs, settings_rs, settings_gpu_memory, settings_mp4_crf,
            settings_keep_temp_png, settings_keep_temp_mp4, settings_keep_temp_json,
            settings_model_name, settings_text_encoder, settings_text_encoder_2, settings_tokenizer, settings_tokenizer_2,
            settings_vae, settings_feature_extractor, settings_image_encoder, settings_transformer
        ],
        outputs=[gr.Markdown()]
    )

    restore_defaults_button.click(
        fn=lambda: restore_original_defaults(return_ips_defaults=True),
        inputs=[],
        outputs=[
            settings_use_teacache, settings_seed, settings_video_length, settings_steps,
            settings_cfg, settings_gs, settings_rs, settings_gpu_memory, settings_mp4_crf,
            settings_keep_temp_png, settings_keep_temp_mp4, settings_keep_temp_json,
            settings_model_name, settings_text_encoder, settings_text_encoder_2, settings_tokenizer, settings_tokenizer_2,
            settings_vae, settings_feature_extractor, settings_image_encoder, settings_transformer
        ]
    )



    # Set default prompt and length
    default_prompt, default_n_prompt, default_length, default_gs, default_steps, default_teacache, default_seed, default_cfg, default_rs, default_gpu_memory, default_mp4_crf, default_keep_temp_png, default_keep_temp_mp4, default_keep_temp_json = get_default_prompt()
    prompt.value = default_prompt
    n_prompt.value = default_n_prompt
    video_length.value = default_length
    gs.value = default_gs
    steps.value = default_steps
    use_teacache.value = default_teacache
    seed.value = default_seed
    cfg.value = default_cfg
    rs.value = default_rs
    gpu_memory_preservation.value = default_gpu_memory
    mp4_crf.value = default_mp4_crf
    keep_temp_png.value = default_keep_temp_png
    keep_temp_mp4.value = default_keep_temp_mp4
    keep_temp_json.value = default_keep_temp_json

    # Connect UI elements
    save_prompt_button.click(
        save_quick_prompt,
        inputs=[prompt, n_prompt, video_length, gs, steps, use_teacache, seed, cfg, rs, gpu_memory_preservation, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json],
        outputs=[prompt, n_prompt, quick_list, video_length, gs, steps, use_teacache, seed, cfg, rs, gpu_memory_preservation, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json],
        queue=False
    )
    delete_prompt_button.click(
        delete_quick_prompt,
        inputs=[quick_list],
        outputs=[prompt, n_prompt, quick_list, video_length, gs, steps, use_teacache, seed, cfg, rs, gpu_memory_preservation, mp4_crf],
        queue=False
    )
    quick_list.change(
        lambda x, current_n_prompt, current_length, current_gs, current_steps, current_teacache, current_seed, current_cfg, current_rs, current_gpu_memory, current_mp4_crf: (
            x, 
            next((item.get('n_prompt', current_n_prompt) for item in quick_prompts if item['prompt'] == x), current_n_prompt),
            next((item.get('length', current_length) for item in quick_prompts if item['prompt'] == x), current_length),
            next((item.get('gs', current_gs) for item in quick_prompts if item['prompt'] == x), current_gs),
            next((item.get('steps', current_steps) for item in quick_prompts if item['prompt'] == x), current_steps),
            next((item.get('use_teacache', current_teacache) for item in quick_prompts if item['prompt'] == x), current_teacache),
            next((item.get('seed', current_seed) for item in quick_prompts if item['prompt'] == x), current_seed),
            next((item.get('cfg', current_cfg) for item in quick_prompts if item['prompt'] == x), current_cfg),
            next((item.get('rs', current_rs) for item in quick_prompts if item['prompt'] == x), current_rs),
            next((item.get('gpu_memory_preservation', current_gpu_memory) for item in quick_prompts if item['prompt'] == x), current_gpu_memory),
            next((item.get('mp4_crf', current_mp4_crf) for item in quick_prompts if item['prompt'] == x), current_mp4_crf)
        ) if x else (x, current_n_prompt, current_length, current_gs, current_steps, current_teacache, current_seed, current_cfg, current_rs, current_gpu_memory, current_mp4_crf),
        inputs=[quick_list, n_prompt, video_length, gs, steps, use_teacache, seed, cfg, rs, gpu_memory_preservation, mp4_crf],
        outputs=[prompt, n_prompt, video_length, gs, steps, use_teacache, seed, cfg, rs, gpu_memory_preservation, mp4_crf],
        queue=False
    )

    # Add JavaScript to set default prompt on page load
    block.load(
        fn=lambda: (default_prompt, default_length, default_gs, default_steps, default_teacache, default_seed, default_cfg, default_rs, default_gpu_memory, default_mp4_crf),
        inputs=None,
        outputs=[prompt, video_length, gs, steps, use_teacache, seed, cfg, rs, gpu_memory_preservation, mp4_crf],
        queue=False
    )

    # Load queue on startup
    block.load(
        fn=lambda: (update_queue_table(), update_queue_display()),
        inputs=None,
        outputs=[queue_table, queue_display],
        queue=False
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
            edit_gpu_memory_preservation,
            edit_steps,
            edit_cfg,
            edit_gs,
            edit_rs,
            edit_mp4_crf,
            edit_keep_temp_png,
            edit_keep_temp_mp4,
            edit_keep_temp_json,
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
        inputs=[edit_job_name, edit_prompt, edit_n_prompt, edit_video_length, edit_seed, edit_use_teacache, edit_gpu_memory_preservation, edit_steps, edit_cfg, edit_gs, edit_rs, edit_mp4_crf, edit_keep_temp_png, edit_keep_temp_mp4, edit_keep_temp_json],
        outputs=[queue_table, queue_display, edit_group]
    )

    cancel_edit_button.click(
        fn=hide_edit_window,
        outputs=[edit_group]
    )

    ips = [input_image, prompt, n_prompt, seed, video_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json]
    start_button.click(
        fn=process,
        inputs=[
            input_image, prompt, n_prompt, seed, video_length,
            latent_window_size, steps, cfg, gs, rs,
            gpu_memory_preservation, use_teacache, mp4_crf,
            keep_temp_png, keep_temp_mp4, keep_temp_json,
            process_state
        ],
        outputs=[
            queue_button, start_button, abort_button,
            preview_image, result_video,
            progress_desc1, progress_bar1,
            progress_desc2, progress_bar2,
            queue_display, queue_table,
            process_state
        ]
    )
    abort_button.click(
        fn=end_process,
        outputs=[queue_table, queue_display, start_button, abort_button, queue_button]
    )
    queue_button.click(
        fn=add_to_queue_handler,
        inputs=[input_image, prompt, n_prompt, video_length, seed, job_name, use_teacache, gpu_memory_preservation, steps, cfg, gs, rs, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json],
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
    config['IPS Defaults'] = {}
    config['Model Settings'] = {}
    config['System Defaults'] = {}
    save_settings(config)

def create_settings_tab():
    """Create the settings tab with all configuration options"""
    with gr.Tab("Settings"):
        with gr.Tab("Global System Defaults"):
            outputs_folder = gr.Textbox(label="Outputs Folder", value=Config.OUTPUTS_FOLDER)
            job_history_folder = gr.Textbox(label="Job History Folder", value=Config.JOB_HISTORY_FOLDER)
            debug_mode = gr.Checkbox(label="Debug Mode", value=Config.DEBUG_MODE)
            keep_completed_jobs = gr.Checkbox(label="Keep Completed Jobs", value=Config.KEEP_COMPLETED_JOBS)
            
            # Model selection dropdowns
            available_models = get_available_models()
            model_name = gr.Dropdown(label="Default Model", choices=available_models['transformer'], value=Config.DEFAULT_MODEL_NAME)
            text_encoder = gr.Dropdown(label="Text Encoder", choices=available_models['text_encoder'], value=Config.DEFAULT_TEXT_ENCODER)
            text_encoder_2 = gr.Dropdown(label="Text Encoder 2", choices=available_models['text_encoder_2'], value=Config.DEFAULT_TEXT_ENCODER_2)
            tokenizer = gr.Dropdown(label="Tokenizer", choices=available_models['tokenizer'], value=Config.DEFAULT_TOKENIZER)
            tokenizer_2 = gr.Dropdown(label="Tokenizer 2", choices=available_models['tokenizer_2'], value=Config.DEFAULT_TOKENIZER_2)
            vae = gr.Dropdown(label="VAE", choices=available_models['vae'], value=Config.DEFAULT_VAE)
            feature_extractor = gr.Dropdown(label="Feature Extractor", choices=available_models['feature_extractor'], value=Config.DEFAULT_FEATURE_EXTRACTOR)
            image_encoder = gr.Dropdown(label="Image Encoder", choices=available_models['image_encoder'], value=Config.DEFAULT_IMAGE_ENCODER)
            transformer = gr.Dropdown(label="Transformer", choices=available_models['transformer'], value=Config.DEFAULT_TRANSFORMER)
            
            save_settings_button = gr.Button("Save Settings")
            restore_defaults_button = gr.Button("Restore Defaults")
            settings_status = gr.Textbox(label="Status", interactive=False)
            
            save_settings_button.click(
                fn=save_settings_from_ui,
                inputs=[
                    outputs_folder, job_history_folder, debug_mode, keep_completed_jobs,
                    model_name, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                    vae, feature_extractor, image_encoder, transformer
                ],
                outputs=[settings_status]
            )
            
            restore_defaults_button.click(
                fn=lambda: restore_original_defaults(return_ips_defaults=False),
                inputs=[],
                outputs=[
                    outputs_folder, job_history_folder, debug_mode, keep_completed_jobs,
                    model_name, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                    vae, feature_extractor, image_encoder, transformer
                ]
            )

def save_ips_defaults_from_ui(use_teacache, seed, video_length, steps, cfg, gs, rs, gpu_memory, mp4_crf,
                           keep_temp_png, keep_temp_mp4, keep_temp_json,
                           model_name, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                           vae, feature_extractor, image_encoder, transformer):
    """Save IPS defaults from UI settings"""
    config = load_settings()
    if 'IPS Defaults' not in config:
        config['IPS Defaults'] = {}
    section = config['IPS Defaults']
    
    # Save IPS defaults
    section['DEFAULT_USE_TEACACHE'] = repr(use_teacache)
    section['DEFAULT_SEED'] = repr(seed)
    section['DEFAULT_VIDEO_LENGTH'] = repr(video_length)
    section['DEFAULT_STEPS'] = repr(steps)
    section['DEFAULT_CFG'] = repr(cfg)
    section['DEFAULT_GS'] = repr(gs)
    section['DEFAULT_RS'] = repr(rs)
    section['DEFAULT_GPU_MEMORY'] = repr(gpu_memory)
    section['DEFAULT_MP4_CRF'] = repr(mp4_crf)
    section['DEFAULT_KEEP_TEMP_PNG'] = repr(keep_temp_png)
    section['DEFAULT_KEEP_TEMP_MP4'] = repr(keep_temp_mp4)
    section['DEFAULT_KEEP_TEMP_JSON'] = repr(keep_temp_json)
    
    # Save model settings
    if 'Model Settings' not in config:
        config['Model Settings'] = {}
    section = config['Model Settings']
    section['DEFAULT_MODEL_NAME'] = repr(model_name)
    section['DEFAULT_TEXT_ENCODER'] = repr(text_encoder)
    section['DEFAULT_TEXT_ENCODER_2'] = repr(text_encoder_2)
    section['DEFAULT_TOKENIZER'] = repr(tokenizer)
    section['DEFAULT_TOKENIZER_2'] = repr(tokenizer_2)
    section['DEFAULT_VAE'] = repr(vae)
    section['DEFAULT_FEATURE_EXTRACTOR'] = repr(feature_extractor)
    section['DEFAULT_IMAGE_ENCODER'] = repr(image_encoder)
    section['DEFAULT_TRANSFORMER'] = repr(transformer)
    
    save_settings(config)
    return "Settings saved successfully!"

def restore_ips_defaults():
    """Restore IPS defaults to original values"""
    config = load_settings()
    if 'IPS Defaults' not in config:
        config['IPS Defaults'] = {}
    section = config['IPS Defaults']
    
    # Restore IPS defaults
    defaults = Config.get_original_defaults()
    for key, value in defaults.items():
        if key.startswith('DEFAULT_'):
            section[key] = repr(value)
    
    save_settings(config)
    return "IPS defaults restored successfully!"

def on_page_load(process_state):
    """Handle page load/refresh by restoring the UI state"""
    state = ProcessState.from_json(process_state)
    debug_print(f"Page load - Restoring state: {state.to_json()}")
    debug_print(f"Page load - Current job queue: {[job.job_name for job in job_queue]}")
    
    # Always update queue display and table
    queue_display_update = update_queue_display()
    queue_table_update = update_queue_table()
    
    if state.is_processing and state.current_job_name:
        debug_print(f"Page load - Found active job: {state.current_job_name}")
        # Find the current job in the queue
        current_job = next((job for job in job_queue if job.job_name == state.current_job_name), None)
        if current_job and current_job.status == "processing":
            debug_print(f"Page load - Restoring UI for processing job: {current_job.job_name}")
            # Job is still processing, restore full UI state
            return (
                gr.update(interactive=True),   # queue_button
                gr.update(interactive=False),  # start_button
                gr.update(interactive=True),   # abort_button
                gr.update(visible=True),#, value=preview),#value=state.last_preview),  # preview_image
                gr.update(visible=True, value=state.current_video),  # result_video
                state.last_progress or "Resuming job...",  # progress_desc
                state.last_progress_html or "",  # progress_bar
                queue_display_update,
                queue_table_update
            )
        else:
            debug_print(f"Page load - Job {state.current_job_name} no longer processing")
    
    debug_print("Page load - No active job found or job no longer processing")
    return (
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(visible=False),
        gr.update(visible=False),
        "",
        "",
        queue_display_update,
        queue_table_update
    )

block.load(
    fn=on_page_load,
    inputs=[process_state],
    outputs=[
        queue_button, start_button, abort_button,
        preview_image, result_video,
        progress_desc1, progress_bar1,
        progress_desc2, progress_bar2,
        queue_display, queue_table
    ]
)

# End of file