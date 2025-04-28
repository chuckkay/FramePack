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
RESET = '\033[0m'

def debug_print(message):
    """Print debug messages in yellow color"""
    if Config.DEBUG_MODE:
        print(f"{YELLOW}[DEBUG] {message}{RESET}")
    
def alert_print(message):
    """ALERT debug messages in red color"""
    print(f"{RED}[DEBUG] {message}{RESET}")

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
    """Save settings from UI to settings.ini"""
    global Config, settings_config
    Config.DEFAULT_USE_TEACACHE = bool(use_teacache)
    Config.DEFAULT_SEED = int(seed)
    Config.DEFAULT_VIDEO_LENGTH = float(video_length)
    Config.DEFAULT_STEPS = int(steps)
    Config.DEFAULT_CFG = float(cfg)
    Config.DEFAULT_GS = float(gs)
    Config.DEFAULT_RS = float(rs)
    Config.DEFAULT_GPU_MEMORY = float(gpu_memory)
    Config.DEFAULT_MP4_CRF = int(mp4_crf)
    Config.DEFAULT_KEEP_TEMP_PNG = bool(keep_temp_png)
    Config.DEFAULT_KEEP_TEMP_MP4 = bool(keep_temp_mp4)
    Config.DEFAULT_KEEP_TEMP_JSON = bool(keep_temp_json)
    Config.to_settings(settings_config)
    return "Settings saved successfully! this does not change any settings for jobs already in the queue, but you can change pending job settings in the edit jobs tab "

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
    """Save the current queue state to JSON file"""
    try:
        jobs = []
        for job in job_queue:
            job_dict = job.to_dict()
            if job_dict is not None:
                jobs.append(job_dict)
        
        file_path = os.path.abspath(QUEUE_FILE)
        with open(file_path, 'w') as f:
            json.dump(jobs, f, indent=2)
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
            
            debug_print(f"Total jobs in the queue: {len(job_queue)}")
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
    job_hex: str  # Changed to string for hex ID
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
                'job_hex': self.job_hex,
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
                job_hex=data['job_hex'],
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

def save_image_to_temp(image: np.ndarray, job_hex: str) -> str:
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
            
        # Create unique filename using hex ID
        filename = f"queue_image_{job_hex}.png"
        filepath = os.path.join(temp_queue_images, filename)
        # Save image
        pil_image.save(filepath)
        return filepath
    except Exception as e:
        alert_print(f"Error saving image: {str(e)}")
        traceback.print_exc()
        return ""

def add_to_queue(prompt, n_prompt, input_image, total_second_length, seed, use_teacache, gpu_memory_preservation, steps, cfg, gs, rs, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json, status="pending"):
    save_queue()
    try:
        # Make sure queue is loaded
        load_queue()
        
        # Generate a unique hex ID for the job
        job_hex = uuid.uuid4().hex[:8]
        # Save image to temp directory and get path
        image_path = save_image_to_temp(input_image, job_hex)
        if not image_path:
            return None
            
        job = QueuedJob(
            prompt=prompt,
            image_path=image_path,
            video_length=total_second_length,
            job_hex=job_hex,
            seed=seed,  # Keep original seed value, including -1
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
        
        # Find the first completed job to insert before
        insert_index = len(job_queue)
        for i, existing_job in enumerate(job_queue):
            if existing_job.status == "completed":
                insert_index = i
                break
        
        # Insert the new job at the found index
        job_queue.insert(insert_index, job)
        save_queue()  # Save immediately after adding
        return job_hex
    except Exception as e:
        alert_print(f"Error adding job to queue: {str(e)}")
        traceback.print_exc()
        return None

def get_next_job():
    """Get the next job from the queue"""
    try:
        # Make sure queue is loaded
        load_queue()
        
        if job_queue:
            job = job_queue.pop(0)  # Remove and return first job
            save_queue()  # Save after removing job
            return job
        save_queue()
        return None
    except Exception as e:
        print(f"Error getting next job: {str(e)}")
        traceback.print_exc()
        return None

def create_missing_image(job_hex: str) -> tuple[str, str]:
    """Create placeholder images for missing queue images and thumbnails"""
    try:
        # Create a white image with MISSING text
        img = Image.new('RGB', (512, 512), color='white')
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 40)
        text = "MISSING"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x = (512 - text_width) // 2
        y = (512 - text_height) // 2
        draw.text((x, y), text, font=font, fill='black')
        
        # Save full size image
        queue_image_path = os.path.join(temp_queue_images, f"queue_image_{job_hex}.png")
        img.save(queue_image_path)
        
        # Create and save thumbnail
        thumb_img = img.resize((200, 200), Image.Resampling.LANCZOS)
        thumb_path = os.path.join(temp_queue_images, f"thumb_{job_hex}.png")
        thumb_img.save(thumb_path)
        
        return queue_image_path, thumb_path
    except Exception as e:
        alert_print(f"Error creating missing image: {str(e)}")
        traceback.print_exc()
        return "", ""

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
                    new_queue_image, new_thumbnail = create_missing_image(job.job_hex)
                    if new_queue_image and new_thumbnail:
                        job.image_path = new_queue_image
                        job.thumbnail = new_thumbnail
                        job.status = "missing"
                        # save_queue()
                elif not job.thumbnail and job.image_path:
                    try:
                        # Load and resize image for thumbnail
                        img = Image.open(job.image_path)
                        width, height = img.size
                        new_height = 200
                        new_width = int((new_height / height) * width)
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        thumb_path = os.path.join(temp_queue_images, f"thumb_{job.job_hex}.png")
                        img.save(thumb_path)
                        job.thumbnail = thumb_path
                        # save_queue()
                    except Exception as e:
                        alert_print(f"Error creating thumbnail: {str(e)}")
                        job.thumbnail = ""

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
        # Only check for missing images if the job is not being deleted
        if job.status != "deleting":
            # Check if both queue image and thumbnail are missing
            queue_image_missing = not os.path.exists(job.image_path) if job.image_path else True
            thumbnail_missing = not os.path.exists(job.thumbnail) if job.thumbnail else True
            
            if queue_image_missing and thumbnail_missing:
                # Create missing placeholder images
                new_queue_image, new_thumbnail = create_missing_image(job.job_hex)
                if new_queue_image and new_thumbnail:
                    job.image_path = new_queue_image
                    job.thumbnail = new_thumbnail
                    job.status = "missing"
            elif not job.thumbnail and job.image_path:
                try:
                    # Load and resize image for thumbnail
                    img = Image.open(job.image_path)
                    width, height = img.size
                    new_height = 200
                    new_width = int((new_height / height) * width)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    thumb_path = os.path.join(temp_queue_images, f"thumb_{job.job_hex}.png")
                    img.save(thumb_path)
                    job.thumbnail = thumb_path
                except Exception as e:
                    alert_print(f"Error creating thumbnail: {str(e)}")
                    job.thumbnail = ""

        # Add job data to display
        if job.thumbnail:
            try:
                # Read the image and convert to base64
                with open(job.thumbnail, "rb") as img_file:
                    import base64
                    img_data = base64.b64encode(img_file.read()).decode()
                img_md = f'<div style="text-align: center; font-size: 0.8em; color: #666; margin-bottom: 5px;">{job.status}</div><div style="text-align: center; font-size: 0.8em; color: #666;">{job.job_hex}</div><div style="text-align: center; font-size: 0.8em; color: #666;">seed: {job.seed}</div><div style="text-align: center; font-size: 0.8em; color: #666;">Length: {job.video_length:.1f}s</div><img src="data:image/png;base64,{img_data}" alt="Input" style="max-width:100px; max-height:100px; display: block; margin: auto; object-fit: contain; transform: scale(0.75); transform-origin: top left;" />'
            except Exception as e:
                alert_print(f"Error converting image to base64: {str(e)}")
                img_md = ""
        else:
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
                debug_print(f"Found job {job.job_hex} with status {job.status}")
                mark_job_pending(job)
                processing_jobs.append(job)
        
        # Remove these jobs from their current positions
        for job in processing_jobs:
            if job in job_queue:
                job_queue.remove(job)
        
        # Add them back at the top in reverse order (so they maintain their relative order)
        for job in reversed(processing_jobs):
            job_queue.insert(0, job)
            debug_print(f"marked previously aborted job as pending and Moved job {job.job_hex} to top of queue")
        
        # Find all failed jobs and move them to top
        failed_jobs = []
        for job in job_queue:
            if job.status == "failed":
                debug_print(f"Found job {job.job_hex} with status {job.status}")
                mark_job_pending(job)
                failed_jobs.append(job)
        
        # Remove these jobs from their current positions
        for job in failed_jobs:
            if job in job_queue:
                job_queue.remove(job)
        
        # Add them back at the top in reverse order (so they maintain their relative order)
        for job in reversed(failed_jobs):
            job_queue.insert(0, job)
            debug_print(f"marked previously failed job as pending and Moved job {job.job_hex} to top of queue")
        
        # Update in-memory queue and save to JSON
        save_queue()
        debug_print(f"{len(processing_jobs)} aborted jobs found and moved to the top as pending")
        debug_print(f"{len(failed_jobs)} failed jobs found and moved to the top as pending")
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



def clean_up_temp_mp4png(job_id: str, outputs_folder: str, keep_temp_png: bool = False, keep_temp_mp4: bool = False, keep_temp_json: bool = False) -> None:
    """
    Deletes all '<job_id>_<n>.mp4' in outputs_folder except the one with the largest n.
    Also deletes the '<job_id>.png' file and '<job_id>.json' file.
    If keep_temp_png is True, no PNG file will be deleted.
    If keep_temp_mp4 is True, no MP4 files will be deleted.
    If keep_temp_json is True, no JSON file will be deleted.
    """
    debug_print(f"clean_up_temp_mp4png called with keep_temp_png={keep_temp_png}, keep_temp_mp4={keep_temp_mp4}, keep_temp_json={keep_temp_json}")
    if keep_temp_png:
        debug_print(f"Keeping temporary PNG file for job {job_id} as requested")
    if keep_temp_mp4:
        debug_print(f"Keeping temporary MP4 files for job {job_id} as requested")
    if keep_temp_json:
        debug_print(f"Keeping temporary JSON file for job {job_id} as requested")

    # Delete the PNG file
    png_path = os.path.join(job_history_folder, f'{job_id}.png')
    try:
        if os.path.exists(png_path) and not keep_temp_png:
            os.remove(png_path)
            debug_print(f"Deleted PNG file: {png_path}")
    except OSError as e:
        alert_print(f"Failed to delete PNG file {png_path}: {e}")

    # Delete the job_id.JSON job file
    json_path = os.path.join(job_history_folder, f'{job_id}.json')
    try:
        if os.path.exists(json_path) and not keep_temp_json:
            os.remove(json_path)
            debug_print(f"Deleted JSON file: {json_path}")
    except OSError as e:
        alert_print(f"Failed to delete JSON file {json_path}: {e}")

    # regex to grab the trailing number
    pattern = re.compile(rf'^{re.escape(job_id)}_(\d+)\.mp4$')
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
    highest_count, _ = max(candidates, key=lambda x: x[0])

    # delete all but the highest
    for count, fname in candidates:
        if count != highest_count and not (keep_temp_mp4 and fname.endswith('.mp4')):
            path = os.path.join(outputs_folder, fname)
            try:
                os.remove(path)
                
            except OSError as e:
                alert_print(f"Failed to delete {fname}: {e}")
    debug_print(f"Deleted old smaller temp videos")

def create_status_thumbnail(image_path, status, border_color, status_text):
    """Create a thumbnail with status-specific border and text"""
    try:
        # Load and resize image for thumbnail
        img = Image.open(image_path)
        width, height = img.size
        new_height = 200
        new_width = int((new_height / height) * width)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Add border
        border_size = 5
        img_with_border = Image.new('RGB', 
            (img.width + border_size*2, img.height + border_size*2), 
            border_color)
        img_with_border.paste(img, (border_size, border_size))
        
        # Add status text
        draw = ImageDraw.Draw(img_with_border)
        # Use smaller font size for RUNNING text
        font_size = 30 if status_text == "RUNNING" else 40
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            try:
                # DejaVuSans ships with Pillow and is usually available
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except (OSError, IOError):
                # Final fallback to a simple built-in bitmap font
                font = ImageFont.load_default()
        text = status_text
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Position text in center
        x = (img_with_border.width - text_width) // 2
        y = (img_with_border.height - text_height) // 2
        
        # Draw text with black outline
        for offset in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            draw.text((x+offset[0], y+offset[1]), text, font=font, fill=(0,0,0))
        draw.text((x, y), text, font=font, fill=(255,255,255))
        
        return img_with_border
    except Exception as e:
        alert_print(f"Error creating status thumbnail: {str(e)}")
        traceback.print_exc()
        return None

def mark_job_processing(job):
    """Mark a job as processing and update its thumbnail with a red border and RUNNING text"""
    try:
        job.status = "processing"
        
        # Delete existing thumbnail if it exists
        if job.thumbnail and os.path.exists(job.thumbnail):
            os.remove(job.thumbnail)
        
        # Create new thumbnail with processing status
        if job.image_path and os.path.exists(job.image_path):
            # Create thumbnail path if it doesn't exist
            if not job.thumbnail:
                job.thumbnail = os.path.join(temp_queue_images, f"thumb_{job.job_hex}.png")
            
            new_thumbnail = create_status_thumbnail(
                job.image_path,
                "processing",
                (255, 0, 0),  # Red color
                "RUNNING"
            )
            if new_thumbnail:
                new_thumbnail.save(job.thumbnail)
        
        # Move job to top of queue
        if job in job_queue:
            job_queue.remove(job)
            job_queue.insert(0, job)
            save_queue()
            
        return update_queue_table(), update_queue_display()
            
    except Exception as e:
        alert_print(f"Error modifying thumbnail: {str(e)}")
        traceback.print_exc()
        return gr.update(), gr.update()

def mark_job_completed(job):
    """Mark a job as completed and update its thumbnail with a green border and DONE text"""
    try:
        job.status = "completed"
        
        # Delete existing thumbnail if it exists
        if job.thumbnail and os.path.exists(job.thumbnail):
            os.remove(job.thumbnail)
        
        # Create new thumbnail with completed status
        if job.image_path and os.path.exists(job.image_path):
            # Create thumbnail path if it doesn't exist
            if not job.thumbnail:
                job.thumbnail = os.path.join(temp_queue_images, f"thumb_{job.job_hex}.png")
            
            new_thumbnail = create_status_thumbnail(
                job.image_path,
                "completed",
                (0, 255, 0),  # Green color
                "DONE"
            )
            if new_thumbnail:
                new_thumbnail.save(job.thumbnail)
        
        # Move job to bottom of queue
        if job in job_queue:
            job_queue.remove(job)
            job_queue.append(job)  # Add to end of queue
            save_queue()
            
        return update_queue_table(), update_queue_display()
            
    except Exception as e:
        alert_print(f"Error modifying thumbnail: {str(e)}")
        traceback.print_exc()
        return gr.update(), gr.update()

def mark_job_failed(job):
    """Mark a job as failed and update its thumbnail with a green border and DONE text"""
    try:
        job.status = "failed"
        
        # Delete existing thumbnail if it exists
        if job.thumbnail and os.path.exists(job.thumbnail):
            os.remove(job.thumbnail)
        
        # Create new thumbnail with failed status
        if job.image_path and os.path.exists(job.image_path):
            # Create thumbnail path if it doesn't exist
            if not job.thumbnail:
                job.thumbnail = os.path.join(temp_queue_images, f"thumb_{job.job_hex}.png")
            
            new_thumbnail = create_status_thumbnail(
                job.image_path,
                "failed",
                (255, 255, 0),  # Yellow color
                "FAILED"
            )
            if new_thumbnail:
                new_thumbnail.save(job.thumbnail)
        
        # Move job to top of queue
        if job in job_queue:
            job_queue.remove(job)
            job_queue.insert(0, job)
            save_queue()
            
        return update_queue_table(), update_queue_display()
            
    except Exception as e:
        alert_print(f"Error modifying thumbnail: {str(e)}")
        traceback.print_exc()
        return gr.update(), gr.update()
        
def mark_job_pending(job):
    """Mark a job as pending and update its thumbnail to a clean version without border or text"""
    try:
        job.status = "pending"
        
        # Delete existing thumbnail if it exists
        if job.thumbnail and os.path.exists(job.thumbnail):
            os.remove(job.thumbnail)
        
        # Create new clean thumbnail
        if job.image_path and os.path.exists(job.image_path):
            # Create thumbnail path if it doesn't exist
            if not job.thumbnail:
                job.thumbnail = os.path.join(temp_queue_images, f"thumb_{job.job_hex}.png")
            
            # Load and resize image for thumbnail
            img = Image.open(job.image_path)
            width, height = img.size
            new_height = 200
            new_width = int((new_height / height) * width)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save clean thumbnail
            img.save(job.thumbnail)
            
        return update_queue_table(), update_queue_display()
            
    except Exception as e:
        alert_print(f"Error modifying thumbnail: {str(e)}")
        traceback.print_exc()
        return gr.update(), gr.update()

@torch.no_grad()
def worker(input_image, prompt, n_prompt, process_seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json):
    debug_print(f"Worker received seed value: {process_seed}")
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()
    job_failed = False  # Flag to track if job actually failed

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

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

        # this section of code not needed i think
        if isinstance(input_image, tuple):
            input_image = input_image[0]
        if isinstance(input_image, str):
            input_image = np.array(Image.open(input_image))
        elif isinstance(input_image, list):
            if isinstance(input_image[0], tuple):
                input_image = np.array(Image.open(input_image[0][0]))
            else:
                input_image = input_image[0]

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        # Save input image with metadata
        metadata = PngInfo()
        metadata.add_text("prompt", prompt)
        metadata.add_text("negative_prompt", n_prompt) 
        metadata.add_text("seed", str(process_seed))  # This will now be the random seed if it was -1
        metadata.add_text("length", str(total_second_length))
        metadata.add_text("latent_window_size", str(latent_window_size))
        metadata.add_text("steps", str(steps))
        metadata.add_text("cfg", str(cfg))
        metadata.add_text("gs", str(gs))
        metadata.add_text("rs", str(rs))
        metadata.add_text("gpu_memory_preservation", str(gpu_memory_preservation))
        metadata.add_text("use_teacache", str(use_teacache))
        metadata.add_text("mp4_crf", str(mp4_crf))

        Image.fromarray(input_image_np).save(os.path.join(job_history_folder, f'{job_id}.png'), pnginfo=metadata)

        # Save job parameters to the job_id.JSON file
        job_params = {
            "prompt": prompt,
            "negative_prompt": n_prompt,
            "seed": process_seed,  # This will now be the random seed if it was -1
            "video_length": total_second_length,
            "latent_window_size": latent_window_size,
            "steps": steps,
            "cfg": cfg,
            "gs": gs,
            "rs": rs,
            "gpu_memory_preservation": gpu_memory_preservation,
            "use_teacache": use_teacache,
            "mp4_crf": mp4_crf
        }
        if keep_temp_json:
            json_path = os.path.join(job_history_folder, f'{job_id}.json')
            with open(json_path, 'w') as f:
                json.dump(job_params, f, indent=2)


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

        rnd = torch.Generator("cpu").manual_seed(process_seed)  
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

            debug_print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

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
                percent_done = (current_time / total_second_length) * 100
                desc = f'Current Job is running, Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {current_time:.2f} seconds of {total_second_length} at (FPS-30). The video is being extended now and is {percent_done:.1f}% done'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return 

            try:
                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
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
            except Exception as e:
                alert_print(f"Error in sampling: {str(e)}")
                traceback.print_exc()
                # If we get an error in sampling, mark the job as failed
                job_failed = True
                raise  # Re-raise to stop processing this job

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
            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                debug_print(f"Calling clean_up_temp_mp4png with keep_temp_png={keep_temp_png}, keep_temp_mp4={keep_temp_mp4}, keep_temp_json={keep_temp_json}")
                clean_up_temp_mp4png(job_id, outputs_folder, keep_temp_png, keep_temp_mp4, keep_temp_json)
                break

    except Exception as e:
        alert_print(f"Error in worker function: {str(e)}")
        traceback.print_exc()
        job_failed = True  # Mark job as failed if we hit an unhandled exception

    finally:
        # Clean up GPU resources
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Find the job in the queue and update its status
        for job in job_queue:
            if job.status == "processing":
                if job_failed:
                    job.status = "failed"
                    mark_job_failed(job)
                else:
                    job.status = "completed"
                    mark_job_completed(job)

                break

        # Save the updated queue
        save_queue()

    stream.output_queue.push(('end', None))
    return



def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json):
    global stream
    
    # Initialize variables
    output_filename = None
    job_hex = None
    process_image = None
    process_prompt = prompt
    process_n_prompt = n_prompt
    process_seed = seed
    process_length = total_second_length
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
    
    # If we have new images, add them to queue first
    if input_image is not None:
        # Convert Gallery tuples to numpy arrays if needed
        if isinstance(input_image, list):
            # Multiple images case
            input_images = [np.array(Image.open(img[0])) for img in input_image]
           
            # Add each image as a separate job with pending status
            for img in input_images:
                job_hex = add_to_queue(
                    prompt=prompt,
                    n_prompt=n_prompt,
                    input_image=img,
                    total_second_length=total_second_length,
                    seed=seed,
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
                if job_hex is not None:
                    # Create thumbnail for the job
                    job = next((job for job in job_queue if job.job_hex == job_hex), None)
                    if job and job.image_path:
                        try:
                            # Load and resize image for thumbnail
                            img = Image.open(job.image_path)
                            width, height = img.size
                            new_height = 200
                            new_width = int((new_height / height) * width)
                            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            thumb_path = os.path.join(temp_queue_images, f"thumb_{job_hex}.png")
                            img.save(thumb_path)
                            job.thumbnail = thumb_path
                            save_queue()
                        except Exception as e:
                            alert_print(f"Error creating thumbnail: {str(e)}")
                            job.thumbnail = ""
        else:
            # Single image case
            input_image = np.array(Image.open(input_image[0]))
            
            # Add single image job
            job_hex = add_to_queue(
                prompt=prompt,
                n_prompt=n_prompt,
                input_image=input_image,
                total_second_length=total_second_length,
                seed=seed,
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
            if job_hex is not None:
                # Create thumbnail for the job
                job = next((job for job in job_queue if job.job_hex == job_hex), None)
                if job and job.image_path:
                    try:
                        # Load and resize image for thumbnail
                        img = Image.open(job.image_path)
                        width, height = img.size
                        new_height = 200
                        new_width = int((new_height / height) * width)
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        thumb_path = os.path.join(temp_queue_images, f"thumb_{job_hex}.png")
                        img.save(thumb_path)
                        job.thumbnail = thumb_path
                        save_queue()
                    except Exception as e:
                        alert_print(f"Error creating thumbnail: {str(e)}")
                        job.thumbnail = ""
        
        # Update queue display after adding new jobs
        update_queue_table()
        update_queue_display()
        
        # Check for pending jobs again after adding new ones
        pending_jobs = [job for job in job_queue if job.status.lower() == "pending"]
    
    if pending_jobs:
        # Process first pending job
        next_job = pending_jobs[0]
        mark_job_processing(next_job)  # Use new function to mark as processing
        save_queue()
        queue_table_update, queue_display_update = mark_job_processing(next_job)  # Use new function to mark as processing
        job_hex = next_job.job_hex
        
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
        debug_print(f"Job {next_job.job_hex} initial seed value: {process_seed}")
        
        # Generate random seed if seed is -1
        if process_seed == -1:
            process_seed = random.randint(0, 2**32 - 1)
            debug_print(f"Generated new random seed for job {next_job.job_hex}: {process_seed}")
            # # Update the job's seed in the queue this is wrong
            # next_job.seed = process_seed
            save_queue()

        process_length = next_job.video_length if hasattr(next_job, 'video_length') else total_second_length
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
        # No input image and no pending jobs
        debug_print("No input image and no pending jobs to process")
        yield (
            None,  # result_video
            None,  # preview_image
            "No input image and no pending jobs to process",  # progress_desc
            '',    # progress_bar
            gr.update(interactive=True),  # start_button
            gr.update(interactive=False),  # end_button
            gr.update(interactive=True),   # queue_button (always enabled)
            update_queue_table(),         # queue_table
            update_queue_display()        # queue_display
        )
        return
    
    # Start processing
    stream = AsyncStream()
    debug_print(f"Starting worker")
    async_run(worker, process_image, process_prompt, process_n_prompt, process_seed, 
             process_length, latent_window_size, process_steps, 
             process_cfg, process_gs, process_rs, 
             process_gpu_memory_preservation, process_teacache, process_mp4_crf, process_keep_temp_png, process_keep_temp_mp4, process_keep_temp_json)
    
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
                        mark_job_completed(job)
                        clean_up_temp_mp4png(job_hex, outputs_folder, keep_temp_png, keep_temp_mp4, keep_temp_json)
                        cleanup_orphaned_files()
                        queue_table_update, queue_display_update = mark_job_completed(job)
                        save_queue()
                        update_queue_table()
                        update_queue_display()

                # Then check if we should continue processing (only if end button wasn't clicked)
                if not stream.input_queue.top() == 'end':
                    # Find next job to process
                    next_job = None
                    
                    # First check for pending jobs
                    pending_jobs = [job for job in job_queue if job.status.lower() == "pending"]
                    if pending_jobs:
                        next_job = pending_jobs[0]

                    if next_job:
                        # Update next job status to processing
                        mark_job_processing(next_job)
                        queue_table_update, queue_display_update = mark_job_processing(next_job)
                        save_queue()
                        update_queue_table()
                        update_queue_display()
                        save_queue()
                        
                        try:
                            next_image = np.array(Image.open(next_job.image_path))
                        except Exception as e:
                            alert_print(f"ERROR loading next image: {str(e)}")
                            traceback.print_exc()
                            raise
                        
                        # Use job parameters with defaults if missing
                        next_prompt = next_job.prompt if hasattr(next_job, 'prompt') else prompt
                        next_n_prompt = next_job.n_prompt if hasattr(next_job, 'n_prompt') else n_prompt
                        ######## fix this
                        process_seed = next_job.seed if hasattr(next_job, 'seed') else seed
                        debug_print(f"Job {next_job.job_hex} initial seed value: {process_seed}")
 
                        # Generate random seed if seed is -1
                        if process_seed == -1:
                            process_seed = random.randint(0, 2**32 - 1)
                            # debug_print(f"Generated new random seed for job {next_job.job_hex}: {process_seed}")
                            save_queue()

                        next_length = next_job.video_length if hasattr(next_job, 'video_length') else total_second_length
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
                        
                        # Process next job
                        async_run(worker, next_image, next_prompt, next_n_prompt, process_seed, 
                                 next_length, latent_window_size, next_steps, 
                                 next_cfg, next_gs, next_rs, 
                                 next_gpu_memory_preservation, next_teacache, next_mp4_crf, next_keep_temp_png, next_keep_temp_mp4, next_keep_temp_json)
                    else:
                        job_queue[:] = [job for job in job_queue if job.status != "completed"]
                        save_queue()
                        # No more jobs, return to initial state
                        cleanup_orphaned_files()

                        yield (
                            output_filename,  # result_video
                            gr.update(visible=False),  # preview_image
                            gr.update(),  # progress_desc
                            '',  # progress_bar
                            gr.update(interactive=True),  # start_button
                            gr.update(interactive=False),  # end_button
                            gr.update(interactive=True),  # queue_button
                            update_queue_table(),         # queue_table
                            update_queue_display()        # queue_display
                        )
                        break
                else:
                    # End button was clicked, stop processing
                    job_queue[:] = [job for job in job_queue if job.status != "completed"]
                    save_queue()
                    yield (
                        output_filename,  # result_video
                        gr.update(visible=False),  # preview_image
                        gr.update(),  # progress_desc
                        '',  # progress_bar
                        gr.update(interactive=True),  # start_button
                        gr.update(interactive=False),  # end_button
                        gr.update(interactive=True),  # queue_button
                        update_queue_table(),         # queue_table
                        update_queue_display()        # queue_display
                    )
                    break
        except KeyboardInterrupt:
            # Handle end button click gracefully
            alert_print("Processing interrupted by user")
            # Find and mark all processing jobs as pending
            for job in job_queue:
                if job.status == "processing":
                    mark_job_pending(job)
            save_queue()
            yield (
                output_filename,  # result_video
                gr.update(visible=False),  # preview_image
                gr.update(),  # progress_desc
                '',  # progress_bar
                gr.update(interactive=True),  # start_button
                gr.update(interactive=False),  # end_button
                gr.update(interactive=True),  # queue_button
                update_queue_table(),         # queue_table
                update_queue_display()        # queue_display
            )
            break
        except Exception as e:
            alert_print(f"Error in process loop: {str(e)}")
            traceback.print_exc()
            # Try to clean up and reset state
            for job in job_queue:
                if job.status == "processing":
                    mark_job_pending(job)
            save_queue()
            yield (
                output_filename,  # result_video
                gr.update(visible=False),  # preview_image
                f"Error occurred: {str(e)}",  # progress_desc
                '',  # progress_bar
                gr.update(interactive=True),  # start_button
                gr.update(interactive=False),  # end_button
                gr.update(interactive=True),  # queue_button
                update_queue_table(),         # queue_table
                update_queue_display()        # queue_display
            )
            break

def end_process():
    """Handle end generation button click - stop all processes and change all processing jobs to pending jobs"""
    try:
        # First send the end signal to stop all processes
        stream.input_queue.push('end')
        
        # Find and update all processing jobs
        jobs_changed = 0
        processing_job = None
        job_queue[:] = [job for job in job_queue if job.status != "completed"]

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
            update_queue_table(),  # queue_table
            update_queue_display(),  # queue_display
            gr.update(interactive=True)  # queue_button (always enabled)
        )
    except Exception as e:
        alert_print(f"Error in end_process: {str(e)}")
        traceback.print_exc()
        return gr.update(), gr.update(), gr.update(interactive=True)  # queue_button (always enabled)

def add_to_queue_handler(input_image, prompt, n_prompt, total_second_length, seed, use_teacache, gpu_memory_preservation, steps, cfg, gs, rs, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json):
    """Handle adding a new job to the queue"""
    if input_image is None or not prompt:
        return gr.update(), gr.update(), gr.update(interactive=True)  # queue_button (always enabled)
    
    try:
        # Convert Gallery tuples to numpy arrays if needed
        if isinstance(input_image, list):
            # Multiple images case
            input_images = [np.array(Image.open(img[0])) for img in input_image]
           
            # Add each image as a separate job with pending status
            for img in input_images:
                job_hex = add_to_queue(
                    prompt=prompt,
                    n_prompt=n_prompt,
                    input_image=img,
                    total_second_length=total_second_length,
                    seed=seed,
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
                if job_hex is not None:
                    # Create thumbnail for the job
                    job = next((job for job in job_queue if job.job_hex == job_hex), None)
                    if job and job.image_path:
                        try:
                            # Load and resize image for thumbnail
                            img = Image.open(job.image_path)
                            width, height = img.size
                            new_height = 200
                            new_width = int((new_height / height) * width)
                            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            thumb_path = os.path.join(temp_queue_images, f"thumb_{job_hex}.png")
                            img.save(thumb_path)
                            job.thumbnail = thumb_path
                            save_queue()
                        except Exception as e:
                            alert_print(f"Error creating thumbnail: {str(e)}")
                            job.thumbnail = ""
        else:
            # Single image case
            input_image = np.array(Image.open(input_image[0]))  # Convert to numpy array
            
            # Add single image job
            job_hex = add_to_queue(
                prompt=prompt,
                n_prompt=n_prompt,
                input_image=input_image,
                total_second_length=total_second_length,
                seed=seed,
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
            if job_hex is not None:
                # Create thumbnail for the job
                job = next((job for job in job_queue if job.job_hex == job_hex), None)
                if job and job.image_path:
                    try:
                        # Load and resize image for thumbnail
                        img = Image.open(job.image_path)
                        width, height = img.size
                        new_height = 200
                        new_width = int((new_height / height) * width)
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        thumb_path = os.path.join(temp_queue_images, f"thumb_{job_hex}.png")
                        img.save(thumb_path)
                        job.thumbnail = thumb_path
                        save_queue()
                    except Exception as e:
                        alert_print(f"Error creating thumbnail: {str(e)}")
                        job.thumbnail = ""
        
        if job_hex is not None:
            save_queue()  # Save after changing statuses
            return update_queue_table(), update_queue_display(), gr.update(interactive=True)  # queue_button (always enabled)
        else:
            return gr.update(), gr.update(), gr.update(interactive=True)  # queue_button (always enabled)
    except Exception as e:
        alert_print(f"Error in add_to_queue_handler: {str(e)}")
        traceback.print_exc()
        return gr.update(), gr.update(), gr.update(interactive=True)  # queue_button (always enabled)


def delete_job(job_hex):
    """Delete a job from the queue and its associated files"""
    try:
        # Find and remove job from queue
        for job in job_queue:
            if job.job_hex == job_hex:
                # Delete associated files
                if os.path.exists(job.image_path):
                    os.remove(job.image_path)
                if os.path.exists(job.thumbnail):
                    os.remove(job.thumbnail)
                job_queue.remove(job)
                break
        save_queue()
        return update_queue_table(), update_queue_display()
    except Exception as e:
        alert_print(f"Error deleting job: {str(e)}")
        traceback.print_exc()
        return update_queue_table(), update_queue_display()

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

def move_job_to_top(job_hex):
    """Move a job to the top of the queue, maintaining processing job at top"""
    try:
        # Find the job's current index
        current_index = None
        for i, job in enumerate(job_queue):
            if job.job_hex == job_hex:
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

def move_job_to_bottom(job_hex):
    """Move a job to the bottom of the queue, maintaining completed jobs at bottom"""
    try:
        # Find the job's current index
        current_index = None
        for i, job in enumerate(job_queue):
            if job.job_hex == job_hex:
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

def move_job(job_hex, direction):
    """Move a job up or down one position in the queue while maintaining sorting rules"""
    try:
        # Find the job's current index
        current_index = None
        for i, job in enumerate(job_queue):
            if job.job_hex == job_hex:
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

def remove_job(job_hex):
    """Delete a job from the queue and its associated files"""
    try:
        # Find and remove job from queue
        for job in job_queue:
            if job.job_hex == job_hex:
                # Delete associated files
                if os.path.exists(job.image_path):
                    os.remove(job.image_path)
                if os.path.exists(job.thumbnail):
                    os.remove(job.thumbnail)
                job_queue.remove(job)
                break
        save_queue()
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
            "",  # job_hex
            gr.update(visible=False)  # edit group visibility
        )
    
    row_index, col_index = evt.index
    button_clicked = evt.value
    
    # Get the job ID from the first column
    job_hex = job_queue[row_index].job_hex
    
    if button_clicked == "⏫️":  # Double up arrow (Top)
        move_job_to_top(job_hex)
    elif button_clicked == "⬆️":  # Single up arrow (Up)
        move_job(job_hex, 'up')
    elif button_clicked == "⬇️":  # Single down arrow (Down)
        move_job(job_hex, 'down')
    elif button_clicked == "⏬️":  # Double down arrow (Bottom)
        move_job_to_bottom(job_hex)
    elif button_clicked == "❌":
        remove_job(job_hex)
    elif button_clicked == "📋":
        copy_job(job_hex)
    elif button_clicked == "✎":
        # Get the job
        job = next((j for j in job_queue if j.job_hex == job_hex), None)
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
                job_hex,
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
        "",  # job_hex
        gr.update(visible=False)  # edit group visibility
    )

def copy_job(job_hex):
    """Create a copy of a job and insert it below the original"""
    try:
        # Find the job
        original_job = next((j for j in job_queue if j.job_hex == job_hex), None)
        if not original_job:
            return update_queue_table(), update_queue_display()
        
        # Create a new job ID
        new_job_hex = uuid.uuid4().hex[:8]
        
        # Copy the image file
        if os.path.exists(original_job.image_path):
            new_image_path = os.path.join(temp_queue_images, f"queue_image_{new_job_hex}.png")
            shutil.copy2(original_job.image_path, new_image_path)
        else:
            new_image_path = ""
        
        # Create new job with copied parameters
        new_job = QueuedJob(
            prompt=original_job.prompt,
            image_path=new_image_path,
            video_length=original_job.video_length,
            job_hex=new_job_hex,
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
            try:
                img = Image.open(new_image_path)
                width, height = img.size
                new_height = 200
                new_width = int((new_height / height) * width)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                thumb_path = os.path.join(temp_queue_images, f"thumb_{new_job_hex}.png")
                img.save(thumb_path)
                new_job.thumbnail = thumb_path
            except Exception as e:
                alert_print(f"Error creating thumbnail: {str(e)}")
                new_job.thumbnail = ""
        
        # Save the updated queue
        save_queue()
        
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


def edit_job(job_hex, new_prompt, new_n_prompt, new_video_length, new_seed, new_use_teacache, new_gpu_memory_preservation, new_steps, new_cfg, new_gs, new_rs, new_mp4_crf, new_keep_temp_png, new_keep_temp_mp4, new_keep_temp_json):
    """Edit a job's parameters"""
    try:
        # Find the job
        for job in job_queue:
            if job.job_hex == job_hex:
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
    return update_queue_table(), update_queue_display()

def delete_pending_jobs():
    """Delete all pending jobs from the queue"""
    global job_queue
    job_queue = [job for job in job_queue if job.status != "pending"]
    save_queue()
    return update_queue_table(), update_queue_display()

def delete_failed_jobs():
    """Delete all failed jobs from the queue"""
    global job_queue
    job_queue = [job for job in job_queue if job.status != "failed"]
    save_queue()
    return update_queue_table(), update_queue_display()

block = gr.Blocks(css=css).queue()

with block:
    gr.Markdown('# FramePack (QueueItUp versionn)')
    
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
                        total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                        latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                        steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                        cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                        gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                        rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                        gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                        mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")
                        keep_temp_png = gr.Checkbox(label="Keep temp PNG file", value=False, info="If checked, temporary PNG file will not be deleted after processing")
                        keep_temp_mp4 = gr.Checkbox(label="Keep temp MP4 files", value=False, info="If checked, temporary MP4 files will not be deleted after processing")
                        keep_temp_json = gr.Checkbox(label="Keep temp JSON file", value=False, info="If checked, temporary JSON file will not be deleted after processing")

                with gr.Column():
                    with gr.Row():
                        start_button = gr.Button(value="Start Generation", interactive=True)
                        end_button = gr.Button(value="End Generation", interactive=False)
                        queue_button = gr.Button(value="Add to Queue", interactive=True)
                    
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
                    edit_job_hex = gr.Textbox(label="Job ID", visible=False)
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

                with gr.Tab("System Defaults"):
                    with gr.Row():
                        with gr.Column():
                            settings_outputs_folder = gr.Textbox(label="Outputs Folder", value=Config.OUTPUTS_FOLDER)
                            settings_job_history_folder = gr.Textbox(label="Job History Folder", value=Config.JOB_HISTORY_FOLDER)
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
    total_second_length.value = default_length
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
        inputs=[prompt, n_prompt, total_second_length, gs, steps, use_teacache, seed, cfg, rs, gpu_memory_preservation, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json],
        outputs=[prompt, n_prompt, quick_list, total_second_length, gs, steps, use_teacache, seed, cfg, rs, gpu_memory_preservation, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json],
        queue=False
    )
    delete_prompt_button.click(
        delete_quick_prompt,
        inputs=[quick_list],
        outputs=[prompt, n_prompt, quick_list, total_second_length, gs, steps, use_teacache, seed, cfg, rs, gpu_memory_preservation, mp4_crf],
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
        inputs=[quick_list, n_prompt, total_second_length, gs, steps, use_teacache, seed, cfg, rs, gpu_memory_preservation, mp4_crf],
        outputs=[prompt, n_prompt, total_second_length, gs, steps, use_teacache, seed, cfg, rs, gpu_memory_preservation, mp4_crf],
        queue=False
    )

    # Add JavaScript to set default prompt on page load
    block.load(
        fn=lambda: (default_prompt, default_length, default_gs, default_steps, default_teacache, default_seed, default_cfg, default_rs, default_gpu_memory, default_mp4_crf),
        inputs=None,
        outputs=[prompt, total_second_length, gs, steps, use_teacache, seed, cfg, rs, gpu_memory_preservation, mp4_crf],
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
            edit_job_hex,
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
            edit_job_hex,
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

    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json]
    start_button.click(
        fn=process, 
        inputs=ips, 
        outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button, queue_button, queue_table, queue_display]
    )
    end_button.click(
        fn=end_process,
        outputs=[queue_table, queue_display, queue_button]
    )
    queue_button.click(
        fn=add_to_queue_handler,
        inputs=[input_image, prompt, n_prompt, total_second_length, seed, use_teacache, gpu_memory_preservation, steps, cfg, gs, rs, mp4_crf, keep_temp_png, keep_temp_mp4, keep_temp_json],
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