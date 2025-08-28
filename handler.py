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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_supabase_client() -> Client:
    supabase_url = os.environ.get("SUPABASE_URL")
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
        
        # AI Toolkit paths (based on diagnostics)
        self.ai_toolkit_path = Path("/app/ai-toolkit")
        self.run_script = self.ai_toolkit_path / "run.py"

    def download_dataset(self, bucket_name: str, dataset_folder: str, local_path: Path):
        logger.info(f"Downloading dataset from {bucket_name}/{dataset_folder}")
        try:
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
            
            logger.info(f"Uploaded {local_file.name} to {bucket_name}/{remote_path}")
            return response
        except Exception as e:
            logger.error(f"Error uploading {local_file}: {str(e)}")
            return None

    def monitor_and_upload_outputs(self, output_dir: Path, bucket_name: str, upload_folder: str):
        uploaded_files = set()
        logger.info(f"Starting file monitoring for {output_dir}")
        
        while True:
            try:
                # Upload samples
                samples_dir = output_dir / "samples"
                if samples_dir.exists():
                    for sample_file in samples_dir.glob("**/*"):
                        if sample_file.is_file() and sample_file not in uploaded_files:
                            relative_path = sample_file.relative_to(output_dir)
                            remote_path = f"{upload_folder}/outputs/{relative_path}"
                            if self.upload_file_to_supabase(sample_file, bucket_name, remote_path):
                                uploaded_files.add(sample_file)
                
                # Upload checkpoints
                for checkpoint in output_dir.glob("*.safetensors"):
                    if checkpoint not in uploaded_files:
                        remote_path = f"{upload_folder}/checkpoints/{checkpoint.name}"
                        if self.upload_file_to_supabase(checkpoint, bucket_name, remote_path):
                            uploaded_files.add(checkpoint)
                
                # Upload logs
                for log_file in output_dir.glob("*.log"):
                    if log_file not in uploaded_files:
                        remote_path = f"{upload_folder}/logs/{log_file.name}"
                        if self.upload_file_to_supabase(log_file, bucket_name, remote_path):
                            uploaded_files.add(log_file)
                
                time.sleep(30)
            except Exception as e:
                logger.error(f"Error in monitoring: {str(e)}")
                time.sleep(60)

    def run_training(self, config_content: str, dataset_config: Dict[str, Any], 
                    upload_config: Dict[str, Any]) -> Dict[str, Any]:
        session_id = None
        try:
            # Generate session ID
            import uuid
            session_id = str(uuid.uuid4())[:8]
            
            # Create session directory
            session_dir = self.training_dir / session_id
            session_dir.mkdir(exist_ok=True)
            
            logger.info(f"Starting training session: {session_id}")
            
            # Download dataset
            dataset_path = session_dir / "dataset"
            self.download_dataset(
                dataset_config["bucket_name"],
                dataset_config["folder_path"],
                dataset_path
            )
            
            # Parse and modify config
            config_data = yaml.safe_load(config_content)
            
            # Update dataset path in config
            if 'config' in config_data and 'process' in config_data['config']:
                for process in config_data['config']['process']:
                    if 'datasets' in process:
                        for i, dataset in enumerate(process['datasets']):
                            process['datasets'][i]['folder_path'] = str(dataset_path)
            
            # Set training folder to session directory
            output_dir = session_dir / "output"
            output_dir.mkdir(exist_ok=True)
            
            if 'config' in config_data and 'process' in config_data['config']:
                for process in config_data['config']['process']:
                    if 'training_folder' in process:
                        process['training_folder'] = str(output_dir)
            
            # Save modified config
            config_file = session_dir / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self.monitor_and_upload_outputs,
                args=(output_dir, upload_config["bucket_name"], upload_config["folder_path"]),
                daemon=True
            )
            monitor_thread.start()
            
            # Run AI Toolkit training
            logger.info("Starting AI Toolkit training...")
            
            # Change to AI Toolkit directory
            original_cwd = os.getcwd()
            os.chdir(str(self.ai_toolkit_path))
            
            try:
                # Command to run AI Toolkit
                cmd = [
                    "python", str(self.run_script),
                    str(config_file)
                ]
                
                logger.info(f"Running command: {' '.join(cmd)}")
                
                # Run the training process
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Stream output
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        logger.info(f"Training: {output.strip()}")
                
                return_code = process.poll()
                
                if return_code == 0:
                    logger.info("Training completed successfully!")
                    
                    # Final upload of any remaining files
                    time.sleep(10)  # Wait for any final file writes
                    
                    return {
                        "success": True,
                        "session_id": session_id,
                        "message": "Training completed successfully",
                        "output_path": str(output_dir)
                    }
                else:
                    logger.error(f"Training failed with return code: {return_code}")
                    return {
                        "success": False,
                        "session_id": session_id,
                        "error": f"Training process failed with return code: {return_code}"
                    }
                    
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            return {
                "success": False,
                "session_id": session_id,
                "error": str(e)
            }

def handler(event):
    input_data = event.get("input", {})
    logger.info(f"Received training request: {json.dumps(input_data, indent=2)}")
    
    # Validate required fields
    required_fields = ["config", "dataset_config", "upload_config"]
    for field in required_fields:
        if field not in input_data:
            error_msg = f"Missing required field: {field}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    # Validate dataset config
    dataset_config = input_data["dataset_config"]
    if not all(k in dataset_config for k in ["bucket_name", "folder_path"]):
        error_msg = "dataset_config must contain 'bucket_name' and 'folder_path'"
        logger.error(error_msg)
        return {"error": error_msg}
    
    # Validate upload config
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
