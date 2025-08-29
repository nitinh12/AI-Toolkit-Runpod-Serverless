import json
import os
import subprocess
import logging
import yaml
import time
import sys
import re
from pathlib import Path
from typing import Dict, Any
import runpod
from supabase import create_client, Client
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging to output everything to stdout/stderr (which goes to worker logs)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Ensure logs go to stdout
    ]
)
logger = logging.getLogger(__name__)

def init_supabase_client() -> Client:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables are required")
    return create_client(supabase_url, supabase_key)

def extract_model_name_from_config(config_content: str) -> str:
    """Extract model name from config content"""
    try:
        config_data = yaml.safe_load(config_content)
        model_name = (
            config_data.get('config', {}).get('name') or
            config_data.get('name') or
            config_data.get('model_name') or
            'default_model'
        )
        return model_name
    except Exception as e:
        logger.warning(f"Could not extract model name from config: {e}")
        return 'default_model'

class FileUploadHandler(FileSystemEventHandler):
    def __init__(self, training_handler, output_dir, bucket_name, upload_folder, model_name):
        self.training_handler = training_handler
        self.output_dir = Path(output_dir)
        self.bucket_name = bucket_name
        self.upload_folder = upload_folder
        self.model_name = model_name
        self.last_upload_times = {}  # Track last upload time for each file
        self.current_step = 0
        self.upload_delay = 3  # seconds to avoid duplicate uploads
        
    def update_current_step(self, step):
        """Update the current training step"""
        self.current_step = step
        
    def can_upload_file(self, file_path):
        """Check if enough time has passed since last upload attempt"""
        now = time.time()
        last_time = self.last_upload_times.get(str(file_path), 0)
        if now - last_time > self.upload_delay:
            self.last_upload_times[str(file_path)] = now
            return True
        return False
        
    def on_created(self, event):
        if not event.is_directory:
            self.handle_file(Path(event.src_path))
    
    def on_modified(self, event):
        if not event.is_directory:
            self.handle_file(Path(event.src_path))
            
    def handle_file(self, file_path):
        try:
            # Small delay to ensure file is fully written
            time.sleep(2)
            
            if not file_path.exists():
                return
                
            # Check if file has meaningful content
            if file_path.stat().st_size == 0:
                return
                
            # Avoid duplicate uploads using time-based filtering
            if not self.can_upload_file(file_path):
                return
                
            if self.should_upload(file_path):
                # Generate proper filename
                new_filename = self.generate_proper_filename(file_path)
                
                if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                    remote_path = f"{self.upload_folder}/{self.model_name}/samples/{new_filename}"
                elif file_path.suffix.lower() == '.safetensors':
                    remote_path = f"{self.upload_folder}/{self.model_name}/checkpoints/{new_filename}"
                elif file_path.name == 'config.yaml':
                    remote_path = f"{self.upload_folder}/{self.model_name}/config/{new_filename}"
                elif file_path.suffix.lower() in ['.log', '.txt']:
                    remote_path = f"{self.upload_folder}/{self.model_name}/logs/{new_filename}"
                else:
                    remote_path = f"{self.upload_folder}/{self.model_name}/outputs/{new_filename}"
                
                if self.training_handler.upload_file_to_supabase(file_path, self.bucket_name, remote_path):
                    logger.info(f"✅ UPLOADED: {file_path.name} -> {new_filename} (step {self.current_step})")
                    
        except Exception as e:
            logger.error(f"❌ UPLOAD ERROR for {file_path}: {str(e)}")
    
    def generate_proper_filename(self, file_path):
        """Generate filename with model name and step number, but keep config.yaml unchanged"""
        if file_path.name == 'config.yaml':
            return 'config.yaml'  # Keep config.yaml unchanged
            
        extension = file_path.suffix.lower()
        
        if extension in ['.png', '.jpg', '.jpeg', '.webp']:
            # Sample images: model-name_step_0001.png
            return f"{self.model_name}_step_{self.current_step:04d}{extension}"
        elif extension == '.safetensors':
            # Checkpoints: model-name_step_0100.safetensors
            return f"{self.model_name}_step_{self.current_step:04d}{extension}"
        else:
            # For other files, keep original name
            return file_path.name
            
    def should_upload(self, file_path):
        upload_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.safetensors', '.log', '.txt', '.yaml', '.yml'}
        return file_path.suffix.lower() in upload_extensions

class TrainingHandler:
    def __init__(self):
        self.supabase = init_supabase_client()
        self.workspace = Path("/workspace")
        self.training_dir = self.workspace / "training"
        self.training_dir.mkdir(exist_ok=True)
        self.file_upload_handler = None
        
        self.ai_toolkit_dir = Path("/app/ai-toolkit")
        self.run_script = self.ai_toolkit_dir / "run.py"
        
        if not self.ai_toolkit_dir.exists() or not self.run_script.exists():
            raise ValueError(f"AI Toolkit not found at expected location: {self.ai_toolkit_dir}")

    def download_dataset(self, bucket_name: str, dataset_folder: str, local_path: Path):
        logger.info(f"📥 Downloading dataset from {bucket_name}/{dataset_folder}")
        try:
            files = self.supabase.storage.from_(bucket_name).list(dataset_folder)
            if not files:
                logger.warning(f"⚠️ No files found in {bucket_name}/{dataset_folder}")
                return

            local_path.mkdir(parents=True, exist_ok=True)
            download_count = 0
            
            for file_info in files:
                if file_info.get('name') and not file_info['name'].endswith('/'):
                    file_path = f"{dataset_folder}/{file_info['name']}"
                    local_file_path = local_path / file_info['name']
                    
                    response = self.supabase.storage.from_(bucket_name).download(file_path)
                    with open(local_file_path, 'wb') as f:
                        f.write(response)
                    download_count += 1
            
            logger.info(f"✅ Dataset download completed: {download_count} files")
        except Exception as e:
            logger.error(f"❌ Error downloading dataset: {str(e)}")
            raise

    def upload_file_to_supabase(self, local_file: Path, bucket_name: str, remote_path: str):
        try:
            with open(local_file, 'rb') as f:
                file_data = f.read()

            try:
                response = self.supabase.storage.from_(bucket_name).upload(
                    remote_path, file_data,
                    file_options={"content-type": "application/octet-stream"}
                )
                return True
            except Exception as upload_error:
                if "already exists" in str(upload_error).lower():
                    response = self.supabase.storage.from_(bucket_name).update(
                        remote_path, file_data,
                        file_options={"content-type": "application/octet-stream"}
                    )
                    return True
                else:
                    raise upload_error

        except Exception as e:
            logger.error(f"❌ Upload error for {local_file}: {str(e)}")
            return False

    def setup_realtime_file_watcher(self, output_dir: Path, bucket_name: str, upload_folder: str, model_name: str):
        logger.info(f"👀 Setting up file monitoring for {output_dir}")
        
        event_handler = FileUploadHandler(self, output_dir, bucket_name, upload_folder, model_name)
        observer = Observer()
        observer.schedule(event_handler, str(output_dir), recursive=True)
        observer.start()
        
        # Store reference to the event handler so we can update step information
        self.file_upload_handler = event_handler
        
        return observer

    def extract_step_from_line(self, line):
        """Extract step number from training output line"""
        # Common patterns for step detection in AI training logs
        step_patterns = [
            r'Step\s*(\d+)',
            r'step\s*(\d+)',
            r'STEP\s*(\d+)',
            r'Epoch\s*\d+.*?Step\s*(\d+)',
            r'(\d+)/\d+\s*\[',
            r'step_(\d+)',
            r'save.*step[_\-](\d+)',
            r'checkpoint.*step[_\-](\d+)',
            r'\bstep:\s*(\d+)',
        ]
        
        for pattern in step_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        return None

    def run_training(self, config_content: str, dataset_config: Dict[str, Any], 
                    upload_config: Dict[str, Any]) -> Dict[str, Any]:
        session_id = None
        file_observer = None
        
        try:
            model_name = extract_model_name_from_config(config_content)
            session_id = f"training_session_{int(time.time())}"
            session_dir = self.training_dir / session_id
            session_dir.mkdir(exist_ok=True)
            
            logger.info(f"🚀 Starting training session {session_id}")
            logger.info(f"📋 Model: {model_name}")
            
            config_file = session_dir / "config.yaml"
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            config_data = yaml.safe_load(config_content)
            
            dataset_path = session_dir / "dataset"
            self.download_dataset(
                dataset_config["bucket_name"],
                dataset_config["folder_path"],
                dataset_path
            )
            
            if 'config' in config_data and 'process' in config_data['config']:
                for process in config_data['config']['process']:
                    if 'datasets' in process:
                        for i, dataset in enumerate(process['datasets']):
                            process['datasets'][i]['folder_path'] = str(dataset_path)
            
            output_dir = session_dir / "output"
            output_dir.mkdir(exist_ok=True)
            
            for process in config_data['config']['process']:
                if 'training_folder' in process:
                    process['training_folder'] = str(output_dir)
            
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            # Setup file monitoring
            file_observer = self.setup_realtime_file_watcher(
                output_dir, 
                upload_config["bucket_name"], 
                upload_config["folder_path"],
                model_name
            )
            
            cmd = [
                "python", str(self.run_script),
                str(config_file)
            ]
            
            logger.info(f"💻 Starting training: {' '.join(cmd)}")
            
            original_cwd = os.getcwd()
            os.chdir(str(self.ai_toolkit_dir))
            
            try:
                # Stream ALL output directly to stdout/stderr in real-time
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                logger.info("✅ Training process started")
                
                # Stream EVERY line and extract step information
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        line = output.rstrip()
                        
                        # Extract step information and update file handler
                        step = self.extract_step_from_line(line)
                        if step is not None and self.file_upload_handler:
                            self.file_upload_handler.update_current_step(step)
                            logger.info(f"📊 Training step updated: {step}")
                        
                        # Print directly to stdout - this goes straight to worker logs
                        print(line, flush=True)
                        sys.stdout.flush()  # Force immediate output
                
                # Get any remaining output
                remaining_output = process.stdout.read()
                if remaining_output:
                    print(remaining_output, flush=True)
                    sys.stdout.flush()
                
                return_code = process.poll()
                
                if return_code == 0:
                    logger.info("🎉 Training completed successfully!")
                    
                    # Wait longer for final file uploads (checkpoints can be large)
                    logger.info("⏳ Waiting for final file uploads...")
                    time.sleep(90)  # Extended wait for final checkpoint
                    
                    return {
                        "success": True,
                        "message": "Training completed successfully",
                        "session_id": session_id,
                        "model_name": model_name,
                        "upload_structure": f"{upload_config['folder_path']}/{model_name}/",
                        "output_path": str(output_dir)
                    }
                else:
                    logger.error(f"❌ Training failed with return code: {return_code}")
                    return {
                        "success": False,
                        "error": f"Training process failed with return code {return_code}",
                        "session_id": session_id
                    }
                    
            finally:
                os.chdir(original_cwd)
                if file_observer:
                    file_observer.stop()
                    file_observer.join()
                    
        except Exception as e:
            logger.error(f"❌ Error in training process: {str(e)}")
            if file_observer:
                file_observer.stop()
                file_observer.join()
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id if session_id else "unknown"
            }

def handler(event):
    input_data = event.get("input", {})
    logger.info(f"📨 Received training request")
    
    required_fields = ["config", "dataset_config", "upload_config"]
    for field in required_fields:
        if field not in input_data:
            error_msg = f"Missing required field: {field}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    dataset_config = input_data["dataset_config"]
    if not all(k in dataset_config for k in ["bucket_name", "folder_path"]):
        error_msg = "dataset_config must contain 'bucket_name' and 'folder_path'"
        logger.error(error_msg)
        return {"error": error_msg}
    
    upload_config = input_data["upload_config"]
    if not all(k in upload_config for k in ["bucket_name", "folder_path"]):
        error_msg = "upload_config must contain 'bucket_name' and 'folder_path'"
        logger.error(error_msg)
        return {"error": error_msg}
    
    try:
        training_handler = TrainingHandler()
        result = training_handler.run_training(
            input_data["config"],
            dataset_config,
            upload_config
        )
        return result
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

if __name__ == "__main__":
    logger.info("🚀 Starting RunPod serverless AI-Toolkit handler...")
    runpod.serverless.start({"handler": handler})
