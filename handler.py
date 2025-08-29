import json
import os
import subprocess
import logging
import yaml
import time
import sys
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
    def __init__(self, training_handler, output_dir, bucket_name, upload_folder):
        self.training_handler = training_handler
        self.output_dir = Path(output_dir)
        self.bucket_name = bucket_name
        self.upload_folder = upload_folder
        self.last_upload_times = {}
        self.upload_delay = 3  # seconds to avoid duplicate uploads
        
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
                
            # Check file has meaningful content
            if file_path.stat().st_size == 0:
                return
            
            # Skip config files
            if file_path.name == 'config.yaml':
                return
                
            # Avoid duplicate uploads using time-based filtering
            if not self.can_upload_file(file_path):
                return
                
            # Calculate relative path from output directory
            try:
                relative_path = file_path.relative_to(self.output_dir)
                remote_path = f"{self.upload_folder}/{relative_path}"
                
                # Convert Windows paths to forward slashes for Supabase
                remote_path = remote_path.replace('\\', '/')
                
                if self.training_handler.upload_file_to_supabase(file_path, self.bucket_name, remote_path):
                    logger.info(f"✅ REAL-TIME UPLOAD: {relative_path}")
                else:
                    logger.warning(f"⚠️ UPLOAD FAILED: {relative_path}")
                    
            except ValueError:
                # File is not within output directory, skip it
                return
                    
        except Exception as e:
            logger.error(f"❌ UPLOAD ERROR for {file_path}: {str(e)}")

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
        """Simple upload without size checking"""
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

    def setup_realtime_file_watcher(self, output_dir: Path, bucket_name: str, upload_folder: str):
        """Set up file watcher for real-time uploads"""
        logger.info(f"👀 Setting up real-time file watcher for {output_dir}")
        
        event_handler = FileUploadHandler(self, output_dir, bucket_name, upload_folder)
        observer = Observer()
        observer.schedule(event_handler, str(output_dir), recursive=True)
        observer.start()
        
        return observer

    def upload_remaining_files(self, output_dir: Path, bucket_name: str, upload_folder: str):
        """Upload any files that might have been missed by the watcher"""
        logger.info(f"📤 Checking for any remaining files to upload...")
        
        uploaded_count = 0
        failed_count = 0
        skipped_count = 0
        
        for file_path in output_dir.rglob('*'):
            if file_path.is_file():
                # Skip config files
                if file_path.name == 'config.yaml':
                    skipped_count += 1
                    continue
                    
                try:
                    relative_path = file_path.relative_to(output_dir)
                    remote_path = f"{upload_folder}/{relative_path}"
                    remote_path = remote_path.replace('\\', '/')
                    
                    # Try to upload (will update if already exists)
                    if self.upload_file_to_supabase(file_path, bucket_name, remote_path):
                        uploaded_count += 1
                        logger.info(f"✅ FINAL UPLOAD: {relative_path}")
                    else:
                        failed_count += 1
                        logger.warning(f"⚠️ FINAL FAILED: {relative_path}")
                        
                except Exception as e:
                    failed_count += 1
                    logger.error(f"❌ ERROR uploading {file_path}: {str(e)}")
        
        if uploaded_count > 0 or failed_count > 0:
            logger.info(f"📊 Final upload check: {uploaded_count} uploaded, {failed_count} failed, {skipped_count} skipped")
        else:
            logger.info(f"📊 No additional files to upload - real-time upload worked perfectly!")
        
        return uploaded_count, failed_count

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
            
            # Download dataset
            dataset_path = session_dir / "dataset"
            self.download_dataset(
                dataset_config["bucket_name"],
                dataset_config["folder_path"],
                dataset_path
            )
            
            # Update config with dataset path
            if 'config' in config_data and 'process' in config_data['config']:
                for process in config_data['config']['process']:
                    if 'datasets' in process:
                        for i, dataset in enumerate(process['datasets']):
                            process['datasets'][i]['folder_path'] = str(dataset_path)
            
            # Set output directory
            output_dir = session_dir / "output"
            output_dir.mkdir(exist_ok=True)
            
            for process in config_data['config']['process']:
                if 'training_folder' in process:
                    process['training_folder'] = str(output_dir)
            
            # Save updated config
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            # Setup real-time file watcher BEFORE starting training
            file_observer = self.setup_realtime_file_watcher(
                output_dir,
                upload_config["bucket_name"],
                upload_config["folder_path"]
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
                
                logger.info("✅ Training process started with real-time file upload monitoring")
                
                # Stream EVERY line to worker logs
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        line = output.rstrip()
                        print(line, flush=True)
                        sys.stdout.flush()
                
                # Get any remaining output
                remaining_output = process.stdout.read()
                if remaining_output:
                    print(remaining_output, flush=True)
                    sys.stdout.flush()
                
                return_code = process.poll()
                
                if return_code == 0:
                    logger.info("🎉 Training completed successfully!")
                    
                    # Wait a bit for any final file operations
                    logger.info("⏳ Waiting for final file operations...")
                    time.sleep(10)
                    
                    # Check for any files that might have been missed
                    uploaded_count, failed_count = self.upload_remaining_files(
                        output_dir,
                        upload_config["bucket_name"],
                        upload_config["folder_path"]
                    )
                    
                    return {
                        "success": True,
                        "message": "Training completed successfully with real-time file uploads",
                        "session_id": session_id,
                        "model_name": model_name,
                        "upload_folder": upload_config["folder_path"],
                        "output_path": str(output_dir),
                        "additional_files_uploaded": uploaded_count,
                        "upload_failures": failed_count
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
