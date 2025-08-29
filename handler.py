import json
import os
import subprocess
import logging
import yaml
import time
from pathlib import Path
from typing import Dict, Any
import runpod
from supabase import create_client, Client
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging to be less verbose but more informative
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        self.uploaded_files = set()
        
    def on_created(self, event):
        if not event.is_directory:
            self.handle_file(Path(event.src_path))
    
    def on_modified(self, event):
        if not event.is_directory:
            self.handle_file(Path(event.src_path))
            
    def handle_file(self, file_path):
        try:
            time.sleep(2)
            
            if file_path in self.uploaded_files:
                return
                
            if not file_path.exists() or file_path.stat().st_size == 0:
                return
                
            if self.should_upload(file_path):
                if file_path.suffix in ['.png', '.jpg', '.jpeg', '.webp']:
                    remote_path = f"{self.upload_folder}/{self.model_name}/samples/{file_path.name}"
                elif file_path.suffix == '.safetensors':
                    remote_path = f"{self.upload_folder}/{self.model_name}/checkpoints/{file_path.name}"
                elif file_path.suffix in ['.log', '.txt']:
                    remote_path = f"{self.upload_folder}/{self.model_name}/logs/{file_path.name}"
                elif file_path.suffix in ['.yaml', '.yml']:
                    remote_path = f"{self.upload_folder}/{self.model_name}/config/{file_path.name}"
                else:
                    remote_path = f"{self.upload_folder}/{self.model_name}/outputs/{file_path.name}"
                
                if self.training_handler.upload_file_to_supabase(file_path, self.bucket_name, remote_path):
                    self.uploaded_files.add(file_path)
                    # Only log successful uploads to reduce log verbosity
                    logger.info(f"✅ UPLOADED: {file_path.name}")
                    
        except Exception as e:
            logger.error(f"❌ UPLOAD ERROR: {str(e)}")
            
    def should_upload(self, file_path):
        upload_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.safetensors', '.log', '.txt', '.yaml', '.yml'}
        return file_path.suffix.lower() in upload_extensions

class TrainingHandler:
    def __init__(self):
        self.supabase = init_supabase_client()
        self.workspace = Path("/workspace")
        self.training_dir = self.workspace / "training"
        self.training_dir.mkdir(exist_ok=True)
        
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
            except Exception as upload_error:
                if "already exists" in str(upload_error).lower():
                    response = self.supabase.storage.from_(bucket_name).update(
                        remote_path, file_data,
                        file_options={"content-type": "application/octet-stream"}
                    )
                else:
                    raise upload_error

            return response
        except Exception as e:
            # Reduce upload error verbosity
            return None

    def setup_realtime_file_watcher(self, output_dir: Path, bucket_name: str, upload_folder: str, model_name: str):
        logger.info(f"👀 Setting up file monitoring for {output_dir}")
        
        event_handler = FileUploadHandler(self, output_dir, bucket_name, upload_folder, model_name)
        observer = Observer()
        observer.schedule(event_handler, str(output_dir), recursive=True)
        observer.start()
        
        return observer

    def create_training_log_file(self, session_dir: Path):
        """Create a dedicated log file for training output"""
        log_file = session_dir / "training_output.log"
        return log_file

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
            
            # Create dedicated log file
            training_log_file = self.create_training_log_file(session_dir)
            
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
                # Write training output to both log file and capture key lines
                with open(training_log_file, 'w') as log_file:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True
                    )
                    
                    logger.info("✅ Training process started")
                    
                    line_count = 0
                    important_lines = []
                    
                    # Process output line by line
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            line = line.rstrip()
                            # Write all output to log file
                            log_file.write(line + '\n')
                            log_file.flush()
                            
                            line_count += 1
                            
                            # Only log important lines to RunPod logs to avoid truncation
                            if any(keyword in line.lower() for keyword in 
                                  ['error', 'warning', 'epoch', 'step', 'loss', 'checkpoint', 'saved', 'completed']):
                                logger.info(f"🔥 {line}")
                                important_lines.append(line)
                            
                            # Log progress every 100 lines
                            if line_count % 100 == 0:
                                logger.info(f"📊 Training progress: {line_count} lines processed")
                    
                    process.wait()
                
                # Upload the complete training log
                log_remote_path = f"{upload_config['folder_path']}/{model_name}/logs/training_output.log"
                self.upload_file_to_supabase(training_log_file, upload_config["bucket_name"], log_remote_path)
                logger.info(f"📋 Complete training log uploaded to Supabase")
                
                if process.returncode == 0:
                    logger.info("🎉 Training completed successfully!")
                    
                    # Wait for final file uploads
                    time.sleep(30)
                    
                    return {
                        "success": True,
                        "message": "Training completed successfully",
                        "session_id": session_id,
                        "model_name": model_name,
                        "upload_structure": f"{upload_config['folder_path']}/{model_name}/",
                        "output_path": str(output_dir),
                        "training_log_uploaded": True,
                        "important_lines": important_lines[-10:] if important_lines else []  # Last 10 important lines
                    }
                else:
                    logger.error(f"❌ Training failed with return code: {process.returncode}")
                    return {
                        "success": False,
                        "error": f"Training process failed with return code {process.returncode}",
                        "session_id": session_id,
                        "training_log_uploaded": True
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

# Streaming handler for progress updates
def handler(event):
    input_data = event.get("input", {})
    
    # Validate required fields
    required_fields = ["config", "dataset_config", "upload_config"]
    for field in required_fields:
        if field not in input_data:
            yield {"error": f"Missing required field: {field}"}
            return
    
    dataset_config = input_data["dataset_config"]
    if not all(k in dataset_config for k in ["bucket_name", "folder_path"]):
        yield {"error": "dataset_config must contain 'bucket_name' and 'folder_path'"}
        return
    
    upload_config = input_data["upload_config"]
    if not all(k in upload_config for k in ["bucket_name", "folder_path"]):
        yield {"error": "upload_config must contain 'bucket_name' and 'folder_path'"}
        return
    
    try:
        # Extract model info for progress updates
        model_name = extract_model_name_from_config(input_data["config"])
        session_id = f"training_session_{int(time.time())}"
        
        # Send initial progress update
        yield {
            "status": "starting",
            "message": f"Starting training for {model_name}",
            "session_id": session_id,
            "model_name": model_name,
            "timestamp": time.time()
        }
        
        # Initialize training handler
        training_handler = TrainingHandler()
        
        yield {
            "status": "initialized",
            "message": "Training handler initialized",
            "timestamp": time.time()
        }
        
        # Run training and get final result
        result = training_handler.run_training(
            input_data["config"],
            dataset_config,
            upload_config
        )
        
        # Send final result
        if result.get("success"):
            yield {
                "status": "completed",
                "message": "Training completed successfully! Complete logs available in Supabase.",
                "result": result,
                "timestamp": time.time()
            }
        else:
            yield {
                "status": "failed", 
                "message": f"Training failed: {result.get('error', 'Unknown error')}",
                "result": result,
                "timestamp": time.time()
            }
            
    except Exception as e:
        yield {
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
        }

if __name__ == "__main__":
    logger.info("🚀 Starting RunPod serverless AI-Toolkit handler...")
    runpod.serverless.start({"handler": handler})
