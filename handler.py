import json
import os
import subprocess
import logging
import yaml
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import runpod
from supabase import create_client, Client
import threading
import time
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_supabase_client() -> Client:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables are required")
    return create_client(supabase_url, supabase_key)

class FileUploadHandler(FileSystemEventHandler):
    def __init__(self, training_handler, output_dir, bucket_name, upload_folder):
        self.training_handler = training_handler
        self.output_dir = Path(output_dir)
        self.bucket_name = bucket_name
        self.upload_folder = upload_folder
        self.uploaded_files = set()
        
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
            
            if file_path in self.uploaded_files:
                return
                
            if not file_path.exists() or file_path.stat().st_size == 0:
                return
                
            # Check if it's a file we want to upload
            if self.should_upload(file_path):
                relative_path = file_path.relative_to(self.output_dir)
                
                # Determine upload path based on file type
                if file_path.suffix in ['.png', '.jpg', '.jpeg', '.webp']:
                    remote_path = f"{self.upload_folder}/samples/{relative_path}"
                elif file_path.suffix == '.safetensors':
                    remote_path = f"{self.upload_folder}/checkpoints/{file_path.name}"
                elif file_path.suffix == '.log':
                    remote_path = f"{self.upload_folder}/logs/{file_path.name}"
                else:
                    remote_path = f"{self.upload_folder}/outputs/{relative_path}"
                
                if self.training_handler.upload_file_to_supabase(file_path, self.bucket_name, remote_path):
                    self.uploaded_files.add(file_path)
                    print(f"🔄 UPLOADED: {file_path.name} -> {remote_path}")  # Print to stdout for client visibility
                    
        except Exception as e:
            print(f"❌ UPLOAD ERROR: {str(e)}")  # Print to stdout for client visibility
            
    def should_upload(self, file_path):
        # Upload images, checkpoints, and logs
        upload_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.safetensors', '.log', '.txt'}
        return file_path.suffix.lower() in upload_extensions

class TrainingHandler:
    def __init__(self):
        self.supabase = init_supabase_client()
        self.workspace = Path("/workspace")
        self.training_dir = self.workspace / "training"
        self.training_dir.mkdir(exist_ok=True)
        
        # Set up AI Toolkit paths based on diagnostics
        self.ai_toolkit_dir = Path("/app/ai-toolkit")
        self.run_script = self.ai_toolkit_dir / "run.py"
        
        # Verify the toolkit is available
        if not self.ai_toolkit_dir.exists() or not self.run_script.exists():
            raise ValueError(f"AI Toolkit not found at expected location: {self.ai_toolkit_dir}")

    def download_dataset(self, bucket_name: str, dataset_folder: str, local_path: Path):
        print(f"📥 DOWNLOADING: Dataset from {bucket_name}/{dataset_folder}")
        try:
            files = self.supabase.storage.from_(bucket_name).list(dataset_folder)
            if not files:
                print(f"⚠️  WARNING: No files found in {bucket_name}/{dataset_folder}")
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
                    print(f"📥 DOWNLOADED: {file_info['name']} ({download_count}/{len(files)})")
            
            print(f"✅ DOWNLOAD COMPLETE: {download_count} files downloaded")
        except Exception as e:
            print(f"❌ DOWNLOAD ERROR: {str(e)}")
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
            print(f"❌ UPLOAD ERROR: {local_file}: {str(e)}")
            return None

    def setup_realtime_file_watcher(self, output_dir: Path, bucket_name: str, upload_folder: str):
        """Set up real-time file monitoring and upload"""
        print(f"👀 MONITORING: Setting up real-time file watcher for {output_dir}")
        
        event_handler = FileUploadHandler(self, output_dir, bucket_name, upload_folder)
        observer = Observer()
        observer.schedule(event_handler, str(output_dir), recursive=True)
        observer.start()
        
        return observer

    def stream_process_output(self, process):
        """Stream process output in real-time to both logs and stdout"""
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.rstrip()
                logger.info(f"TRAINING: {line}")
                # Print to stdout so it gets sent back to client
                print(f"🔥 TRAINING: {line}")
                sys.stdout.flush()  # Ensure immediate output
            else:
                break

    def run_training(self, config_content: str, dataset_config: Dict[str, Any], 
                    upload_config: Dict[str, Any]) -> Dict[str, Any]:
        session_id = None
        file_observer = None
        
        try:
            # Create session directory
            session_id = f"training_session_{int(time.time())}"
            session_dir = self.training_dir / session_id
            session_dir.mkdir(exist_ok=True)
            
            print(f"🚀 STARTING: Training session {session_id}")
            
            # Create config file
            config_file = session_dir / "config.yaml"
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            # Parse config to get dataset path and update it
            config_data = yaml.safe_load(config_content)
            
            # Download dataset
            dataset_path = session_dir / "dataset"
            self.download_dataset(
                dataset_config["bucket_name"],
                dataset_config["folder_path"],
                dataset_path
            )
            
            # Update config with correct dataset path
            if 'config' in config_data and 'process' in config_data['config']:
                for process in config_data['config']['process']:
                    if 'datasets' in process:
                        for i, dataset in enumerate(process['datasets']):
                            process['datasets'][i]['folder_path'] = str(dataset_path)
            
            # Update output path to session directory
            output_dir = session_dir / "output"
            output_dir.mkdir(exist_ok=True)
            
            for process in config_data['config']['process']:
                if 'training_folder' in process:
                    process['training_folder'] = str(output_dir)
            
            # Write updated config
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            print(f"📁 CONFIG: {config_file}")
            print(f"📁 DATASET: {dataset_path}")
            print(f"📁 OUTPUT: {output_dir}")
            
            # Set up real-time file monitoring BEFORE starting training
            file_observer = self.setup_realtime_file_watcher(
                output_dir, 
                upload_config["bucket_name"], 
                upload_config["folder_path"]
            )
            
            # Run AI Toolkit training using the correct command structure
            cmd = [
                "python", str(self.run_script),
                str(config_file)
            ]
            
            print(f"💻 COMMAND: {' '.join(cmd)}")
            
            # Change to AI Toolkit directory for execution
            original_cwd = os.getcwd()
            os.chdir(str(self.ai_toolkit_dir))
            
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                print("✅ PROCESS: Training process started successfully")
                
                # Stream output in real-time
                self.stream_process_output(process)
                
                process.wait()
                
                if process.returncode == 0:
                    print("🎉 SUCCESS: Training completed successfully!")
                    
                    # Wait a bit for final files to be processed
                    time.sleep(30)
                    
                    return {
                        "success": True,
                        "message": "Training completed successfully",
                        "session_id": session_id,
                        "output_path": str(output_dir)
                    }
                else:
                    print(f"❌ FAILED: Training failed with return code: {process.returncode}")
                    return {
                        "success": False,
                        "error": f"Training process failed with return code {process.returncode}",
                        "session_id": session_id
                    }
                    
            finally:
                os.chdir(original_cwd)
                if file_observer:
                    file_observer.stop()
                    file_observer.join()
                    
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            if file_observer:
                file_observer.stop()
                file_observer.join()
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }

def handler(event):
    input_data = event.get("input", {})
    print(f"📨 REQUEST: Received training request")
    
    # Validate required fields
    required_fields = ["config", "dataset_config", "upload_config"]
    for field in required_fields:
        if field not in input_data:
            error_msg = f"Missing required field: {field}"
            print(f"❌ VALIDATION ERROR: {error_msg}")
            return {"error": error_msg}
    
    dataset_config = input_data["dataset_config"]
    if not all(k in dataset_config for k in ["bucket_name", "folder_path"]):
        error_msg = "dataset_config must contain 'bucket_name' and 'folder_path'"
        print(f"❌ VALIDATION ERROR: {error_msg}")
        return {"error": error_msg}
    
    upload_config = input_data["upload_config"]
    if not all(k in upload_config for k in ["bucket_name", "folder_path"]):
        error_msg = "upload_config must contain 'bucket_name' and 'folder_path'"
        print(f"❌ VALIDATION ERROR: {error_msg}")
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
        print(f"❌ HANDLER ERROR: {error_msg}")
        return {"error": error_msg}

if __name__ == "__main__":
    print("🚀 STARTING: RunPod serverless AI-Toolkit handler...")
    runpod.serverless.start({"handler": handler})
