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

# Path to settings file
SETTINGS_FILE = os.path.join(os.getcwd(), 'settings.ini')

# Path to the quick prompts JSON file
PROMPT_FILE = os.path.join(os.getcwd(), 'quick_prompts.json')

# Queue file path
QUEUE_FILE = os.path.join(os.getcwd(), 'job_queue.json')

# Model cache directory
MODEL_CACHE_DIR = os.path.join(os.getcwd(), 'model_cache')
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Model version tracking file
MODEL_VERSIONS_FILE = os.path.join(MODEL_CACHE_DIR, 'model_versions.json')

# Set Hugging Face cache directory
os.environ['HF_HOME'] = MODEL_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = os.path.join(MODEL_CACHE_DIR, 'transformers')
os.environ['HF_DATASETS_CACHE'] = os.path.join(MODEL_CACHE_DIR, 'datasets')

# Temp directory for queue images
temp_queue_images = os.path.join(os.getcwd(), 'temp_queue_images')
os.makedirs(temp_queue_images, exist_ok=True)

# ANSI color codes
YELLOW = '\033[93m'
RED = '\033[31m'
GREEN = '\033[92m'
RESET = '\033[0m'

def debug_print(message):
    """Print debug messages in yellow color"""
    if Config.DEBUG_MODE:
        print(f"{YELLOW}[DEBUG] {message}{RESET}")
    
def alert_print(message):
    """ALERT debug messages in red color"""
    print(f"{RED}[DEBUG] {message}{RESET}")

def info_print(message):
    """Print info messages in green color"""
    print(f"{GREEN}[INFO] {message}{RESET}")

def get_model_versions():
    """Get current versions of models from Hugging Face"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        models = {
            "HunyuanVideo": "hunyuanvideo-community/HunyuanVideo",
            "FluxRedux": "lllyasviel/flux_redux_bfl",
            "FramePack": "lllyasviel/FramePackI2V_HY"
        }
        
        versions = {}
        for name, repo_id in models.items():
            try:
                # Get the latest commit hash
                refs = api.list_repo_refs(repo_id)  # Removed timeout parameter
                if refs and refs.branches:
                    # Get the main/master branch commit
                    main_branch = next((b for b in refs.branches if b.name in ['main', 'master']), None)
                    if main_branch:
                        versions[name] = main_branch.target_commit
            except Exception as e:
                debug_print(f"Could not get version for {name}: {str(e)}")
                versions[name] = None
        
        return versions
    except Exception as e:
        debug_print(f"Could not check model versions: {str(e)}")
        return None

def load_local_versions():
    """Load locally stored model versions"""
    try:
        if os.path.exists(MODEL_VERSIONS_FILE):
            with open(MODEL_VERSIONS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        alert_print(f"Error loading local versions: {str(e)}")
        return {}

def save_local_versions(versions):
    """Save model versions to local file"""
    try:
        with open(MODEL_VERSIONS_FILE, 'w') as f:
            json.dump(versions, f, indent=2)
    except Exception as e:
        alert_print(f"Error saving local versions: {str(e)}")

def check_model_files():
    """Check if required model files are downloaded and up to date"""
    model_paths = {
        "HunyuanVideo": os.path.join(MODEL_CACHE_DIR, "models--hunyuanvideo-community--HunyuanVideo"),
        "FluxRedux": os.path.join(MODEL_CACHE_DIR, "models--lllyasviel--flux_redux_bfl"),
        "FramePack": os.path.join(MODEL_CACHE_DIR, "models--lllyasviel--FramePackI2V_HY")
    }
    
    # First check if all required models exist locally
    missing_models = []
    for name, path in model_paths.items():
        if not os.path.exists(path):
            missing_models.append(name)
    
    # If we have all models locally, try to check for updates
    if not missing_models:
        # Get current versions from Hugging Face
        current_versions = get_model_versions()
        
        # If we couldn't get versions (no internet/slow), use local versions
        if not current_versions:
            info_print(f"\nUsing local model versions (could not check for updates)")
            return
        
        # Load local versions
        local_versions = load_local_versions()
        
        # Check each model
        outdated_models = []
        for name in model_paths.keys():
            if name in current_versions and name in local_versions:
                if current_versions[name] != local_versions[name]:
                    outdated_models.append(name)
        
        if outdated_models:
            print(f"\nThe following models have updates available:")
            for model in outdated_models:
                print(f"- {model}")
            print("\nThese models will be updated to their latest versions.")
        else:
            info_print(f"\nAll models are up to date in {MODEL_CACHE_DIR}")
        
        # Save current versions
        save_local_versions(current_versions)
    else:
        # If we're missing models, we need to try to download them
        print(f"\nThe following models will be downloaded to {MODEL_CACHE_DIR}:")
        for model in missing_models:
            print(f"- {model}")
        print("\nThis is a one-time download. Future runs will use the local files.")
        
        # Try to get versions for the download
        current_versions = get_model_versions()
        if current_versions:
            save_local_versions(current_versions)

def save_settings(config):
    """Save settings to settings.ini file"""
    with open(SETTINGS_FILE, 'w') as f:
        config.write(f)

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

    # System defaults
    OUTPUTS_FOLDER: str = None
    JOB_HISTORY_FOLDER: str = None
    DEBUG_MODE: bool = None

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
            'OUTPUTS_FOLDER': './outputs/',
            'JOB_HISTORY_FOLDER': './job_history/',
            'DEBUG_MODE': False
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

        # Save System Defaults section
        if 'System Defaults' not in config:
            config['System Defaults'] = {}
        section = config['System Defaults']
        system_defaults = ['OUTPUTS_FOLDER', 'JOB_HISTORY_FOLDER', 'DEBUG_MODE']
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

def save_settings_from_ui(use_teacache, seed, video_length, steps, cfg, gs, rs, gpu_memory, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json):
    """Save settings from UI to settings.ini and update Config class values"""
    try:
        config = load_settings()
        section = config['IPS Defaults']
        
        # Update settings.ini
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
        
        # Save to file
        save_settings(config)
        
        # Update Config class values in memory
        Config.DEFAULT_USE_TEACACHE = use_teacache
        Config.DEFAULT_SEED = seed
        Config.DEFAULT_VIDEO_LENGTH = video_length
        Config.DEFAULT_STEPS = steps
        Config.DEFAULT_CFG = cfg
        Config.DEFAULT_GS = gs
        Config.DEFAULT_RS = rs
        Config.DEFAULT_GPU_MEMORY = gpu_memory
        Config.DEFAULT_MP4_CRF = mp4_crf
        Config.DEFAULT_KEEP_TEMP_PNG = keep_temp_png
        Config.DEFAULT_KEEP_TEMP_MP4 = keep_temp_mp4
        Config.DEFAULT_KEEP_TEMP_JSON = keep_temp_json
        
        debug_print("Settings saved successfully")
        return "Settings saved successfully! This does not change any settings for jobs already in the queue, but you can change pending job settings in the edit jobs tab."
    except Exception as e:
        alert_print(f"Error saving settings: {str(e)}")
        return f"Error saving settings: {str(e)}"

def restore_original_defaults():
    """Restore original default values"""
    global Config, settings_config
    defaults = Config.get_original_defaults()
    for key, value in defaults.items():
        setattr(Config, key, value)
    Config.to_settings(settings_config)
    return (
        Config.DEFAULT_USE_TEACACHE,
        Config.DEFAULT_SEED,
        Config.DEFAULT_VIDEO_LENGTH,
        Config.DEFAULT_STEPS,
        Config.DEFAULT_CFG,
        Config.DEFAULT_GS,
        Config.DEFAULT_RS,
        Config.DEFAULT_GPU_MEMORY,
        Config.DEFAULT_MP4_CRF,
        Config.DEFAULT_KEEP_TEMP_PNG,
        Config.DEFAULT_KEEP_TEMP_MP4,
        Config.DEFAULT_KEEP_TEMP_JSON
    )

def save_system_settings_from_ui(outputs_folder, job_history_folder, debug_mode):
    """Save system settings from UI to settings.ini"""
    global Config, settings_config
    
    # Create folders if they don't exist
    os.makedirs(outputs_folder, exist_ok=True)
    os.makedirs(job_history_folder, exist_ok=True)
    
    Config.OUTPUTS_FOLDER = outputs_folder
    Config.JOB_HISTORY_FOLDER = job_history_folder
    Config.DEBUG_MODE = bool(debug_mode)
    Config.to_settings(settings_config)
    
    # Update local variables
    setup_local_variables()
    
    return "System settings saved successfully! some settings may require restart to work"

def restore_system_defaults():
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
    Config.to_settings(settings_config)
    
    # Update local variables
    setup_local_variables()
    
    return Config.OUTPUTS_FOLDER, Config.JOB_HISTORY_FOLDER, Config.DEBUG_MODE

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
    global job_history_folder, outputs_folder, debug_mode
    job_history_folder = Config.JOB_HISTORY_FOLDER
    outputs_folder = Config.OUTPUTS_FOLDER
    debug_mode = Config.DEBUG_MODE

# Initialize settings
settings_config = load_settings()
Config = Config.from_settings(settings_config)

# Create necessary directories using values from Config
os.makedirs(Config.OUTPUTS_FOLDER, exist_ok=True)
os.makedirs(Config.JOB_HISTORY_FOLDER, exist_ok=True)

# Check for model files before loading
check_model_files()

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
        if not job_name:  # This will catch both None and empty string
            # Generate a hex ID for the job name
            hex_id = uuid.uuid4().hex[:8]  # Get first 8 characters of UUID hex
            job_name = f"job_{hex_id}"
        else:
            # If job_name is provided, append hex ID
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
            job_queue.append(job)
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
            job.thumbnail = create_thumbnail(job, status_change=False)
            job_queue.append(job)
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
    status_overlay = "RUNNING" if job.status == "processing" else job.status.upper()
    
    try:
    # Try to load arial font, fall back to default if not available
        font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    try:
       # Handle text-to-video case (job.image_path is text2video)
        if job.image_path == "text2video":
            debug_print("in create_thumbnail job.image_path is text2video")
            if not job.thumbnail or status_change:  # Create new thumbnail if none exists or status changed
                debug_print("in create_thumbnail text2video creating new thumbnail")
                # Create a text-to-video thumbnail
                img = Image.new('RGB', (200, 200), color='black')
                draw = ImageDraw.Draw(img)
                # Add text-to-video indicator
                draw.text((100, 80), "Text to Video", fill='white', anchor="mm", font=font)
                draw.text((100, 100), "Generation", fill='white', anchor="mm", font=font)
                draw.text((100, 120), status_overlay, fill=status_color, anchor="mm", font=small_font)
                
                # Save thumbnail
                thumbnail_path = os.path.join(temp_queue_images, f"thumb_{job.job_name}.png")
                debug_print(f"try to save thumbnail to {thumbnail_path}")
                img.save(thumbnail_path)
                debug_print(f"thumbnail saved to {thumbnail_path}")
                job.thumbnail = thumbnail_path
                save_queue()
            return job.thumbnail

        # Handle missing image-based cases 
        if job.image_path != "text2video" and not os.path.exists(job.image_path) and not job.thumbnail:
            # Create missing image thumbnail
            img = Image.new('RGB', (200, 200), color='white')
            draw = ImageDraw.Draw(img)
            draw.text((100, 100), "MISSING IMAGE", fill='red', anchor="mm", font=font)
            # Save thumbnail
            thumbnail_path = os.path.join(temp_queue_images, f"thumb_{job.job_name}.png")
            img.save(thumbnail_path)
            job.thumbnail = thumbnail_path
            save_queue()
            return thumbnail_path

        # Normal case - create thumbnail from existing image
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
            draw.text((100, 100), status_overlay, fill='white', anchor="mm", font=font)
        
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
                
                if queue_image_missing and thumbnail_missing:
                    # Create missing placeholder images
                    new_thumbnail = create_thumbnail(job, status_change=False)
                elif not job.thumbnail and job.image_path:
                    job.thumbnail = create_thumbnail(job, status_change=False)

            # Add job data to display
            if job.thumbnail:
                caption = f"{job.status}\n\nPrompt: {job.prompt} \n\n Negative: {job.n_prompt}\n\nLength: {job.video_length}s\nGS: {job.gs}"
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
    try:
        # First load the queue from JSON
        load_queue()
        
        ## change this code to only delete if keep_job_history is false (create variable keep_job_history default is true
        # Count completed jobs before removal
        completed_jobs_count = len([job for job in job_queue if job.status == "completed"])
        
        # Now remove completed jobs
        job_queue[:] = [job for job in job_queue if job.status != "completed"]
        
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
            debug_print(f"marked previously failed job as pending and Moved job {job.job_name} to top of queue")
        
        # Update in-memory queue and save to JSON
        save_queue()
        debug_print(f"{len(processing_jobs)} aborted jobs found and moved to the top as pending")
        debug_print(f"{len(failed_jobs)} failed jobs found and moved to the top as pending")
        debug_print(f"Total jobs in the queue:{len(job_queue)}")
    except Exception as e:
        alert_print(f"Error in reset_processing_jobs: {str(e)}")
        traceback.print_exc()

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

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

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

stream = AsyncStream()



def clean_up_temp_mp4png(job):
    job_name = job.job_name
    """
    Deletes all '<job_name>_<n>.mp4' in outputs_folder except the one with the largest n.
    Also deletes the '<job_name>.png' file and '<job_name>.json' file.
    Uses the keep_temp settings from the job object to determine which files to keep.
    """
    

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
    """Mark a job as processing and update its thumbnail"""
    try:
        job.status = "processing"
        
        # Delete existing thumbnail if it exists
        if job.thumbnail and os.path.exists(job.thumbnail):
            os.remove(job.thumbnail)
        
        # For text-to-video jobs, create a new processing thumbnail
        if job.image_path == "text2video":
            job.thumbnail = create_thumbnail(job, status_change=True)
        else:
            # For image-based jobs, update the existing thumbnail
            job.thumbnail = create_thumbnail(job, status_change=True)
        
        # Move job to top of queue
        if job in job_queue:
            job_queue.remove(job)
            job_queue.insert(0, job)
            
        save_queue()
        return update_queue_table(), update_queue_display()
    except Exception as e:
        alert_print(f"Error marking job as processing: {str(e)}")
        traceback.print_exc()
        return gr.update(), gr.update()

def mark_job_completed(job):
    """Mark a job as completed and update its thumbnail"""
    try:
        job.status = "completed"
        
        # Delete existing thumbnail if it exists
        if job.thumbnail and os.path.exists(job.thumbnail):
            os.remove(job.thumbnail)
        
        # Create new thumbnail with completed status
        if job.image_path:
            job.thumbnail = create_thumbnail(job, status_change=True)
        
        # Move job to bottom of queue
        if job in job_queue:
            job_queue.remove(job)
            job_queue.append(job)
            save_queue()
            
        return update_queue_table(), update_queue_display()
            
    except Exception as e:
        alert_print(f"Error marking job as completed: {str(e)}")
        traceback.print_exc()
        return gr.update(), gr.update()

def mark_job_failed(job):
    """Mark a job as failed and update its thumbnail"""
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
    """Mark a job as pending and update its thumbnail"""
    try:
        job.status = "pending"
        
        # Delete existing thumbnail if it exists
        if job.thumbnail and os.path.exists(job.thumbnail):
            os.remove(job.thumbnail)
        
        # Create new clean thumbnail
        if job.image_path:
            job.thumbnail = create_thumbnail(job, status_change=True)
            
        return update_queue_table(), update_queue_display()
            
    except Exception as e:
        alert_print(f"Error marking job as pending: {str(e)}")
        traceback.print_exc()
        return gr.update(), gr.update()

@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, job_name, video_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json, next_job):
    """Worker function to process a job"""
    debug_print(f"Starting worker for job {job_name}")

    total_latent_sections = (video_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_failed = None
    job_id = generate_timestamp() #not used

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))
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

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

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

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))


        # Handle text-to-video case
        if input_image is None:
            # Create a blank image for text-to-video with default resolution
            default_resolution = 640  # Default resolution for text-to-video
            input_image_np = np.zeros((default_resolution, default_resolution, 3), dtype=np.uint8)
            height = width = default_resolution
            # Skip saving the blank image since it's not needed for text-to-video
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

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

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

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
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

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                current_time = max(0, (total_generated_latent_frames * 4 - 3) / 30)
                percent_done = (current_time / video_length) * 100
                desc = f'Current Job is running, Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {current_time:.2f} seconds of {video_length} at (FPS-30). The video is being extended now and is {percent_done:.1f}% done'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
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

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_name}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            # Find the job that's currently being processed

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                break
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
    debug_print(f"last section Calling clean_up_temp_mp4png")
    clean_up_temp_mp4png(next_job)


    if next_job.image_path == "text2video":
        mp4_path = os.path.join(outputs_folder, f'{next_job.job_name}.mp4')
        if os.path.exists(mp4_path):
            import cv2

            cap = cv2.VideoCapture(mp4_path)
            # Seek to the 30th frame (zero-based index 29)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 29)
            ret, frame = cap.read()
            if ret:
                # Resize to 200×200
                thumb = cv2.resize(frame, (200, 200), interpolation=cv2.INTER_AREA)

                # Overlay centered yellow "DONE" text
                text = "DONE"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                thickness = 2
                color = (0, 255, 255)  # BGR for yellow

                # Calculate text size to center it
                (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
                x = (thumb.shape[1] - text_w) // 2
                y = (thumb.shape[0] + text_h) // 2

                cv2.putText(
                    thumb,
                    text,
                    (x, y),
                    font,
                    scale,
                    color,
                    thickness,
                    cv2.LINE_AA
                )

                thumb_path = os.path.join(temp_queue_images, f'thumb_{next_job.job_name}.png')
                queue_path = os.path.join(temp_queue_images, f'queue_image_{next_job.job_name}.png')
                cv2.imwrite(thumb_path, thumb)
                # cv2.imwrite(queue_path, thumb)
                next_job.thumbnail = thumb_path

            cap.release()

    mark_job_completed(next_job)
    stream.output_queue.push(('end', None))
    return


def process(input_image, prompt, n_prompt, seed, video_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json):
    global stream
    is_processing = True  # Set processing state to True when starting
    save_queue()  # Save the processing state
    
    # Initialize variables
    job_name = None
    process_image = None
    process_prompt = prompt
    process_n_prompt = n_prompt
    process_seed = seed
    process_job_name = job_name
    process_length = video_length
    process_steps = steps
    process_cfg = cfg
    process_gs = gs
    process_rs = rs
    process_gpu_memory_preservation = gpu_memory_preservation
    process_teacache = use_teacache
    process_keep_temp_png = keep_temp_png
    process_keep_temp_mp4 = keep_temp_mp4
    process_keep_temp_json = keep_temp_json
    process_mp4_crf = mp4_crf
    
    # First check for pending jobs
    pending_jobs = [job for job in job_queue if job.status.lower() == "pending"]

    if pending_jobs:
        # Process first pending job
        next_job = pending_jobs[0]

        
        mark_job_processing(next_job)
        save_queue()
        queue_table_update, queue_display_update = mark_job_processing(next_job)
        job_name = next_job.job_name
        
        # Handle NULL image path (text-to-video)
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
    else:
        # No pending jobs
        debug_print("No pending jobs to process")
        yield (
            None,  # result_video
            None,  # preview_image
            "No input image and no pending jobs to process",  # progress_desc
            '',    # progress_bar
            gr.update(interactive=True),   # start_button
            gr.update(interactive=False),  # end_button
            gr.update(interactive=True),   # queue_button (always enabled)
            update_queue_table(),         # queue_table
            update_queue_display()        # queue_display
        )
        return
    
    # Start processing
    stream = AsyncStream()
    debug_print(f"Starting worker for job {next_job.job_name}")


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



    async_run(worker, process_image, process_prompt, process_n_prompt, seed, process_job_name, 
             process_length, latent_window_size, process_steps, 
             process_cfg, process_gs, process_rs, 
             process_gpu_memory_preservation, process_teacache, process_mp4_crf, process_keep_temp_png, process_keep_temp_mp4, process_keep_temp_json,next_job)

    # Initial yield with updated queue display and button states
    yield (
        None,  # result_video
        None,  # preview_image
        '',    # progress_desc
        '',    # progress_bar
        gr.update(interactive=False),  # start_button
        gr.update(interactive=True),   # end_button
        gr.update(interactive=True),   # queue_button (always enabled)
        update_queue_table(),         # queue_table
        update_queue_display()        # queue_display
    )

    # Process output queue
    while True:
        try:
            flag, data = stream.output_queue.next()

            if flag == 'file':
                output_filename = data
                yield (
                    output_filename,  # result_video
                    gr.update(),  # preview_image
                    gr.update(),  # progress_desc
                    gr.update(),  # progress_bar
                    gr.update(interactive=False),  # start_button
                    gr.update(interactive=True),   # end_button
                    gr.update(interactive=True),   # queue_button (always enabled)
                    update_queue_table(),         # queue_table
                    update_queue_display()        # queue_display
                )

            if flag == 'progress':
                preview, desc, html = data
                yield (
                    gr.update(),  # result_video
                    gr.update(visible=True, value=preview),  # preview_image
                    desc,  # progress_desc
                    html,  # progress_bar
                    gr.update(interactive=False),  # start_button
                    gr.update(interactive=True),   # end_button
                    gr.update(interactive=True),   # queue_button (always enabled)
                    update_queue_table(),         # queue_table
                    update_queue_display()        # queue_display
                )

            if flag == 'end':
                # Find the current processing job
                for job in job_queue:
                    if job.status == "processing":
                        queue_table_update, queue_display_update = mark_job_completed(job)
                        save_queue()
                        update_queue_table()
                        update_queue_display()

                # Check if we should continue processing (only if end button wasn't clicked)
                if not stream.input_queue.top() == 'end':
                    # Find next pending job
                    next_job = None
                    pending_jobs = [job for job in job_queue if job.status.lower() == "pending"]
                    if pending_jobs:
                        next_job = pending_jobs[0]
                    if next_job:
                       

                            
                        mark_job_processing(next_job)
                        queue_table_update, queue_display_update = mark_job_processing(next_job)
                        save_queue()
                        update_queue_table()
                        update_queue_display()
                        save_queue()
                        
                        # Handle NULL image path (text-to-video)
                        if next_job.image_path == "text2video":
                            next_image = None
                        else:
                            try:
                                next_image = np.array(Image.open(next_job.image_path))
                            except Exception as e:
                                alert_print(f"ERROR loading image: {str(e)}")
                                traceback.print_exc()
                                raise

                        # Use job parameters with defaults if missing
                        next_prompt = next_job.prompt if hasattr(next_job, 'prompt') else prompt
                        next_n_prompt = next_job.n_prompt if hasattr(next_job, 'n_prompt') else n_prompt
                        next_seed = next_job.seed if hasattr(next_job, 'seed') else seed
                        debug_print(f"Job {next_job.job_name} initial seed value: {process_seed}")
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
                        
                        # Start processing next job
                        stream = AsyncStream()
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
                    else:
                        debug_print("No more pending jobs to process")
                        yield (
                            None,  # result_video
                            None,  # preview_image
                            "No pending jobs to process",  # progress_desc
                            '',    # progress_bar
                            gr.update(interactive=True),   # "Add to Queue" button
                            gr.update(interactive=False),  # "Start Queued Jobs" button
                            gr.update(interactive=True),   # "Abort Generation" button
                            update_queue_table(),         # queue_table
                            update_queue_display()        # queue_display
                        )
                        return
                else:
                    debug_print("End button clicked, stopping processing")
                    yield (
                        None,  # result_video
                        None,  # preview_image
                        "Processing stopped",  # progress_desc
                        '',    # progress_bar
                        gr.update(interactive=True),   # "Add to Queue" button
                        gr.update(interactive=False),  # "Start Queued Jobs" button
                        gr.update(interactive=True),   # "Abort Generation" button
                        update_queue_table(),         # queue_table
                        update_queue_display()        # queue_display
                    )
                    return

        except Exception as e:
            alert_print(f"Error in process loop: {str(e)}")
            traceback.print_exc()
            yield (
                None,  # result_video
                None,  # preview_image
                f"Error: {str(e)}",  # progress_desc
                '',    # progress_bar
                gr.update(interactive=True),   # "Add to Queue" button
                gr.update(interactive=False),  # "Start Queued Jobs" button
                gr.update(interactive=True),   # "Abort Generation" button
                update_queue_table(),         # queue_table
                update_queue_display()        # queue_display
            )
            return

def end_process():
    """Handle end generation button click - stop all processes and change all processing jobs to pending jobs"""
    try:
        # First send the end signal to stop all processes
        stream.input_queue.push('end')

        
        # Find and update all processing jobs
        jobs_changed = 0
        processing_job = None
        # job_queue[:] = [job for job in job_queue if job.status != "completed"]

        # First find the processing job
        for job in job_queue:
            if job.status == "processing":
                processing_job = job
                break
        # Then process all jobs
        for job in job_queue:
            if job.status == "processing":
                mark_job_pending(job)  # Use new function to mark as pending
                save_queue()
                queue_table_update, queue_display_update = mark_job_pending(job)  # Use new function to mark as pending
                jobs_changed += 1
        
        # If we found a processing job, move it to the top
        if processing_job:
            job_queue.remove(processing_job)
            job_queue.insert(0, processing_job)
        
        save_queue()
        return (
            update_queue_table(),         # queue_table
            update_queue_display(),       # queue_display
            gr.update(interactive=True),  # start_button
            gr.update(interactive=False), # end_button
            gr.update(interactive=True)   # queue_button (always enabled)
        )
    except Exception as e:
        alert_print(f"Error in end_process: {str(e)}")
        traceback.print_exc()
        return (
            gr.update(),                  # queue_table
            gr.update(),                  # queue_display
            gr.update(interactive=True),  # start_button
            gr.update(interactive=False), # end_button
            gr.update(interactive=True)   # queue_button (always enabled)
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
                gr.update(interactive=False),  # end_button
                gr.update(interactive=True),   # queue_button (always enabled)
                update_queue_table(),         # queue_table
                update_queue_display()        # queue_display
            )

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
            for img_tuple in input_image:
                input_image = np.array(Image.open(img_tuple[0]))  # Convert to numpy array
                
                # Add job for each image
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
        else:  # direction == 'down'
            # Find the next non-completed job
            new_index = current_index + 1
            while new_index < len(job_queue) and job_queue[new_index].status == "completed":
                new_index += 1
            if new_index >= len(job_queue):
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
        if job and job.status == "pending":
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
            
        # Create a new job ID by keeping the prefix and replacing the last 8 chars
        prefix = original_job.job_name[:-8] if len(original_job.job_name) > 8 else ""
        new_job_name = prefix + uuid.uuid4().hex[:8]
        
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
.queue-gallery .gallery-item {
    margin: 5px;
}

/* Hide table headers */
.gradio-dataframe thead {
    display: none !important;
}

/* Prevent Gradio's wrapper from centering the grid */
.gradio-dataframe > div {
    display: flex !important;
    align-items: flex-start !important;
}

/* Force AG-Grid to fill its container */
.ag-theme-gradio .ag-root-wrapper {
    height: 100% !important;
}
.ag-theme-gradio .ag-body-viewport,
.ag-theme-gradio .ag-center-cols-viewport {
    height: 100% !important;
}

/* Set each row to 100px tall */
.ag-theme-gradio .ag-body-viewport .ag-center-cols-container .ag-row {
    height: 100px !important;
}

/* Vertically center the prompt text in column 7 */
.ag-theme-gradio .ag-center-cols-container .ag-row .ag-cell:nth-child(7) {
    display: flex !important;
    align-items: center !important;
    padding: 0 8px !important;
}

/* First column fixed width */
.gradio-dataframe th:first-child,
.gradio-dataframe td:first-child {
    width: 150px !important;
    min-width: 150px !important;
}

/* Remove orange selection highlight */
.gradio-dataframe td.selected,
.gradio-dataframe td:focus,
.gradio-dataframe tr.selected td,
.gradio-dataframe tr:focus td {
    background-color: transparent !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Style the arrow buttons */
.gradio-dataframe td:nth-child(2),
.gradio-dataframe td:nth-child(3),
.gradio-dataframe td:nth-child(4),
.gradio-dataframe td:nth-child(5),
.gradio-dataframe td:nth-child(6),
.gradio-dataframe th:nth-child(2),
.gradio-dataframe th:nth-child(3),
.gradio-dataframe th:nth-child(4),
.gradio-dataframe th:nth-child(5),
.gradio-dataframe th:nth-child(6) {
    cursor: pointer;
    color: #666;
    font-weight: bold;
    transition: color 0.2s;
    width: 42px !important;
    min-width: 42px !important;
    max-width: 42px !important;
    text-align: center !important;
    # vertical-align: middle !important;
    # font-size: 1.5em !important;
    padding: 0 !important;
    # overflow: visible !important;
}
.gradio-dataframe td:nth-child(2):hover,
.gradio-dataframe td:nth-child(3):hover,
.gradio-dataframe td:nth-child(4):hover,
.gradio-dataframe td:nth-child(5):hover,
.gradio-dataframe td:nth-child(6):hover {
    color: #000;
}

/* Align all headers */
.gradio-dataframe th {
    text-align: center !important;
    padding: 8px !important;
}
.output-html:last-of-type, .gradio-html:last-of-type, .gradio-html-block:last-of-type {
    display: none !important;
}
/* Column‐width overrides */
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
                # Only allow editing if job is pending
                if job.status not in ("pending", "completed"):
                    return update_queue_table(), update_queue_display(), gr.update(visible=False)

                # Update job parameters
                job.prompt = new_prompt
                job.n_prompt = new_n_prompt
                job.status = "pending"
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

block = gr.Blocks(css=css).queue()

with block:
    gr.Markdown('# FramePack (QueueItUp version)')
    
    with gr.Tabs():
        with gr.Tab("Framepack_QueueItUp"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Gallery(
                        label="Image",
                        height=320,
                        columns=3,
                        object_fit="contain"
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
                        queue_button = gr.Button(value="Add to Queue", interactive=True)
                        start_button = gr.Button(value="Start Queued Jobs", interactive=True)
                        end_button = gr.Button(value="Abort Generation", interactive=False)
                    
                    preview_image = gr.Image(label="Next Latents", height=200, visible=False)
                    result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
                    gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
                    progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                    progress_bar = gr.HTML('', elem_classes='no-generating-animation')


                    queue_display = gr.Gallery(
                        label="Job Queue Gallery",
                        show_label=True,
                        columns=3,
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
                        save_edit_button = gr.Button("Save Changes")
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
                elem_classes=["gradio-dataframe"]
            )


        with gr.Tab("Settings"):
            with gr.Tabs():
                with gr.Tab("Job Defaults"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### this will set the new job defaults, REQUIRES RESTART, these changes will not change setting for jobs that are already in the queue")
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
                            
                            with gr.Row():
                                save_defaults_button = gr.Button("Save job settings as Defaults")
                                restore_defaults_button = gr.Button("Restore Original job settings")

                with gr.Tab("Global System Defaults"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### this will set the new system defaults, it REQUIRES RESTART to take")
                            settings_outputs_folder = gr.Textbox(label="Outputs Folder this is where your videos will be saved", value=Config.OUTPUTS_FOLDER)
                            settings_job_history_folder = gr.Textbox(label="Job History Folder this is where the job settings json file and job input image is stored", value=Config.JOB_HISTORY_FOLDER)
                            settings_debug_mode = gr.Checkbox(label="Debug Mode", value=Config.DEBUG_MODE)
                            
                            with gr.Row():
                                save_system_defaults_button = gr.Button("Save System Settings")
                                restore_system_defaults_button = gr.Button("Restore System Defaults")

    # Connect settings buttons and all other UI event bindings at the top level (not in a nested with block)
    save_defaults_button.click(
        fn=save_settings_from_ui,
        inputs=[
            settings_use_teacache, settings_seed, settings_video_length, settings_steps,
            settings_cfg, settings_gs, settings_rs, settings_gpu_memory, settings_mp4_crf,
            settings_keep_temp_png, settings_keep_temp_mp4, settings_keep_temp_json
        ],
        outputs=[gr.Markdown()]
    )

    restore_defaults_button.click(
        fn=restore_original_defaults,
        inputs=[],
        outputs=[
            settings_use_teacache, settings_seed, settings_video_length, settings_steps,
            settings_cfg, settings_gs, settings_rs, settings_gpu_memory, settings_mp4_crf,
            settings_keep_temp_png, settings_keep_temp_mp4, settings_keep_temp_json
        ]
    )

    save_system_defaults_button.click(
        fn=save_system_settings_from_ui,
        inputs=[
            settings_outputs_folder,
            settings_job_history_folder,
            settings_debug_mode
        ],
        outputs=[gr.Markdown()]
    )

    restore_system_defaults_button.click(
        fn=restore_system_defaults,
        inputs=[],
        outputs=[
            settings_outputs_folder,
            settings_job_history_folder,
            settings_debug_mode
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
        inputs=[
            edit_job_name,
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
            edit_keep_temp_json
        ],
        outputs=[queue_table, queue_display, edit_group]
    )

    ips = [input_image, prompt, n_prompt, seed, video_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json]
    start_button.click(
        fn=process, 
        inputs=ips, 
        outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button, queue_button, queue_table, queue_display]
    )
    end_button.click(
        fn=end_process,
        outputs=[queue_table, queue_display, start_button, end_button, queue_button]
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