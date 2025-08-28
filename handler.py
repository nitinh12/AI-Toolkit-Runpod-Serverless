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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Supabase client
def init_supabase_client() -> Client:
    supabase_url = os.environ.get("SUPABASE_URL")
    # Changed from SUPABASE_ANON_KEY to SUPABASE_SERVICE_KEY for private bucket access
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables are required")
    
    return create_client(supabase_url, supabase_key)

class TrainingHandler:
    def __init__(self):
        self.supabase = init_supabase_client()
        self.workspace = Path("/workspace")
        self.training_dir = self.workspace / "training"
        self.training_dir.mkdir(exist_ok=True)
        
    def download_dataset(self, bucket_name: str, dataset_folder: str, local_path: Path):
        """Download dataset from Supabase storage"""
        logger.info(f"Downloading dataset from {bucket_name}/{dataset_folder}")
        
        try:
            # List all files in the dataset folder
            files = self.supabase.storage.from_(bucket_name).list(dataset_folder)
            
            if not files:
                logger.warning(f"No files found in {bucket_name}/{dataset_folder}")
                return
            
            local_path.mkdir(parents=True, exist_ok=True)
            
            download_count = 0
            for file_info in files:
                if file_info.get('name') and not file_info['name'].endswith('/'):
                    file_path = f"{dataset_folder}/{file_info['name']}"
                    local_file_path = local_path / file_info['name']
                    
                    # Download file
                    response = self.supabase.storage.from_(bucket_name).download(file_path)
                    
                    with open(local_file_path, 'wb') as f:
                        f.write(response)
                    
                    download_count += 1
                    logger.info(f"Downloaded: {file_info['name']} ({download_count}/{len(files)})")
            
            logger.info(f"Dataset download completed: {download_count} files downloaded")
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            raise
    
    def upload_file_to_supabase(self, local_file: Path, bucket_name: str, remote_path: str):
        """Upload a file to Supabase storage"""
        try:
            with open(local_file, 'rb') as f:
                file_data = f.read()
            
            # Try to upload, if file exists, update it
            try:
                response = self.supabase.storage.from_(bucket_name).upload(
                    remote_path, file_data, 
                    file_options={"content-type": "application/octet-stream"}
                )
            except Exception as upload_error:
                # If upload fails (file might exist), try update
                if "already exists" in str(upload_error).lower():
                    response = self.supabase.storage.from_(bucket_name).update(
                        remote_path, file_data,
                        file_options={"content-type": "application/octet-stream"}
                    )
                else:
                    raise upload_error
            
            logger.info(f"Uploaded {local_file.name} to {bucket_name}/{remote_path}")
            return response
            
        except Exception as e:
            logger.error(f"Error uploading {local_file}: {str(e)}")
            return None
    
    def monitor_and_upload_outputs(self, output_dir: Path, bucket_name: str, upload_folder: str):
        """Monitor output directory and upload new files"""
        uploaded_files = set()
        
        logger.info(f"Starting file monitoring for {output_dir}")
        
        while True:
            try:
                # Check for new sample images
                samples_dir = output_dir / "samples"
                if samples_dir.exists():
                    for sample_file in samples_dir.glob("**/*"):
                        if sample_file.is_file() and sample_file not in uploaded_files:
                            relative_path = sample_file.relative_to(output_dir)
                            remote_path = f"{upload_folder}/outputs/{relative_path}"
                            
                            if self.upload_file_to_supabase(sample_file, bucket_name, remote_path):
                                uploaded_files.add(sample_file)
                
                # Check for checkpoints
                for checkpoint in output_dir.glob("*.safetensors"):
                    if checkpoint not in uploaded_files:
                        remote_path = f"{upload_folder}/checkpoints/{checkpoint.name}"
                        if self.upload_file_to_supabase(checkpoint, bucket_name, remote_path):
                            uploaded_files.add(checkpoint)
                
                # Check for logs
                for log_file in output_dir.glob("*.log"):
                    if log_file not in uploaded_files:
                        remote_path = f"{upload_folder}/logs/{log_file.name}"
                        if self.upload_file_to_supabase(log_file, bucket_name, remote_path):
                            uploaded_files.add(log_file)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def run_training(self, config_content: str, dataset_config: Dict[str, Any], 
                    upload_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run AI toolkit training with monitoring and uploads"""
        
        session_id = None
        try:
            # Create unique training session directory
            session_id = f"training_{int(time.time())}"
            session_dir = self.training_dir / session_id
            session_dir.mkdir(exist_ok=True)
            
            logger.info(f"Created training session: {session_id}")
            
            # Download dataset
            dataset_local_path = session_dir / "dataset"
            self.download_dataset(
                dataset_config['bucket_name'],
                dataset_config['folder_path'],
                dataset_local_path
            )
            
            # Parse and modify config
            config = yaml.safe_load(config_content)
            
            # Update config with local dataset path
            if 'config' in config and 'process' in config['config']:
                for process_item in config['config']['process']:
                    if 'datasets' in process_item:
                        for dataset in process_item['datasets']:
                            if 'folder_path' in dataset:
                                dataset['folder_path'] = str(dataset_local_path)
            
            # Set output directory
            output_dir = session_dir / "outputs"
            output_dir.mkdir(exist_ok=True)
            
            # Update save_dir in config
            if 'config' in config and 'process' in config['config']:
                for process_item in config['config']['process']:
                    if 'save' in process_item:
                        process_item['save']['save_dir'] = str(output_dir)
                    else:
                        process_item['save'] = {'save_dir': str(output_dir)}
            
            # Write modified config
            config_file = session_dir / "training_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Starting training with config: {config_file}")
            logger.info(f"Dataset path: {dataset_local_path}")
            logger.info(f"Output directory: {output_dir}")
            
            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self.monitor_and_upload_outputs,
                args=(output_dir, upload_config['bucket_name'], upload_config['folder_path']),
                daemon=True
            )
            monitor_thread.start()
            
            # Run training
            cmd = [
                "python", "-m", "toolkit.train",
                "--config", str(config_file)
            ]
            
            logger.info(f"Executing command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output and log
            output_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    output_lines.append(line)
                    logger.info(f"TRAINING: {line}")
            
            return_code = process.poll()
            
            # Final upload of any remaining files
            logger.info("Training completed, uploading final files...")
            time.sleep(10)  # Give monitor thread time to catch up
            
            # Upload final config
            self.upload_file_to_supabase(
                config_file, 
                upload_config['bucket_name'], 
                f"{upload_config['folder_path']}/config/training_config.yaml"
            )
            
            result = {
                "success": return_code == 0,
                "return_code": return_code,
                "session_id": session_id,
                "output_lines": output_lines[-50:],  # Last 50 lines to avoid huge responses
                "total_output_lines": len(output_lines),
                "output_dir": str(output_dir),
                "uploaded_to": f"{upload_config['bucket_name']}/{upload_config['folder_path']}"
            }
            
            logger.info(f"Training completed with return code: {return_code}")
            return result
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id if session_id else None
            }

def handler(event):
    """RunPod serverless handler"""
    input_data = event.get("input", {})
    
    # Log incoming request
    logger.info(f"Received training request: {json.dumps(input_data, indent=2)}")
    
    required_fields = ["config", "dataset_config", "upload_config"]
    for field in required_fields:
        if field not in input_data:
            error_msg = f"Missing required field: {field}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    # Validate dataset_config
    dataset_config = input_data["dataset_config"]
    if not all(k in dataset_config for k in ["bucket_name", "folder_path"]):
        error_msg = "dataset_config must contain 'bucket_name' and 'folder_path'"
        logger.error(error_msg)
        return {"error": error_msg}
    
    # Validate upload_config
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
    logger.info("Starting RunPod serverless AI-Toolkit handler...")
    runpod.serverless.start({"handler": handler})
