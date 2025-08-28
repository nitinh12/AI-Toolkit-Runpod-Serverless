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
    supabase_key = os.environ.get("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY environment variables are required")
    
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
            
            local_path.mkdir(parents=True, exist_ok=True)
            
            for file_info in files:
                if file_info.get('name') and not file_info['name'].endswith('/'):
                    file_path = f"{dataset_folder}/{file_info['name']}"
                    local_file_path = local_path / file_info['name']
                    
                    # Download file
                    response = self.supabase.storage.from_(bucket_name).download(file_path)
                    
                    with open(local_file_path, 'wb') as f:
                        f.write(response)
                    
                    logger.info(f"Downloaded: {file_info['name']}")
                    
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            raise
    
    def upload_file_to_supabase(self, local_file: Path, bucket_name: str, remote_path: str):
        """Upload a file to Supabase storage"""
        try:
            with open(local_file, 'rb') as f:
                file_data = f.read()
            
            response = self.supabase.storage.from_(bucket_name).upload(
                remote_path, file_data, 
                file_options={"content-type": "application/octet-stream"}
            )
            
            logger.info(f"Uploaded {local_file.name} to {bucket_name}/{remote_path}")
            return response
            
        except Exception as e:
            logger.error(f"Error uploading {local_file}: {str(e)}")
            return None
    
    def monitor_and_upload_outputs(self, output_dir: Path, bucket_name: str, upload_folder: str):
        """Monitor output directory and upload new files"""
        uploaded_files = set()
        
        while True:
            try:
                # Check for new sample images
                samples_dir = output_dir / "samples"
                if samples_dir.exists():
                    for sample_file in samples_dir.glob("**/*"):
                        if sample_file.is_file() and sample_file not in uploaded_files:
                            relative_path = sample_file.relative_to(output_dir)
                            remote_path = f"{upload_folder}/outputs/{relative_path}"
                            
                            self.upload_file_to_supabase(sample_file, bucket_name, remote_path)
                            uploaded_files.add(sample_file)
                
                # Check for checkpoints
                for checkpoint in output_dir.glob("*.safetensors"):
                    if checkpoint not in uploaded_files:
                        remote_path = f"{upload_folder}/checkpoints/{checkpoint.name}"
                        self.upload_file_to_supabase(checkpoint, bucket_name, remote_path)
                        uploaded_files.add(checkpoint)
                
                # Check for logs
                for log_file in output_dir.glob("*.log"):
                    if log_file not in uploaded_files:
                        remote_path = f"{upload_folder}/logs/{log_file.name}"
                        self.upload_file_to_supabase(log_file, bucket_name, remote_path)
                        uploaded_files.add(log_file)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def run_training(self, config_content: str, dataset_config: Dict[str, Any], 
                    upload_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run AI toolkit training with monitoring and uploads"""
        
        try:
            # Create unique training session directory
            session_id = f"training_{int(time.time())}"
            session_dir = self.training_dir / session_id
            session_dir.mkdir(exist_ok=True)
            
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
            if 'datasets' in config:
                for dataset in config['datasets']:
                    if 'folder_path' in dataset:
                        dataset['folder_path'] = str(dataset_local_path)
            
            # Set output directory
            output_dir = session_dir / "outputs"
            output_dir.mkdir(exist_ok=True)
            config['save_dir'] = str(output_dir)
            
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
                    output_lines.append(output.strip())
                    logger.info(f"TRAINING: {output.strip()}")
            
            return_code = process.poll()
            
            # Final upload of any remaining files
            time.sleep(10)  # Give monitor thread time to catch up
            
            # Upload final config and logs
            self.upload_file_to_supabase(
                config_file, 
                upload_config['bucket_name'], 
                f"{upload_config['folder_path']}/config/training_config.yaml"
            )
            
            result = {
                "success": return_code == 0,
                "return_code": return_code,
                "session_id": session_id,
                "output_lines": output_lines,
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
                "session_id": session_id if 'session_id' in locals() else None
            }

def handler(event):
    """RunPod serverless handler"""
    input_data = event.get("input", {})
    
    required_fields = ["config", "dataset_config", "upload_config"]
    for field in required_fields:
        if field not in input_data:
            return {"error": f"Missing required field: {field}"}
    
    # Validate dataset_config
    dataset_config = input_data["dataset_config"]
    if not all(k in dataset_config for k in ["bucket_name", "folder_path"]):
        return {"error": "dataset_config must contain 'bucket_name' and 'folder_path'"}
    
    # Validate upload_config
    upload_config = input_data["upload_config"]
    if not all(k in upload_config for k in ["bucket_name", "folder_path"]):
        return {"error": "upload_config must contain 'bucket_name' and 'folder_path'"}
    
    try:
        training_handler = TrainingHandler()
        result = training_handler.run_training(
            input_data["config"],
            dataset_config,
            upload_config
        )
        return result
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
