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

    def upload_output_folder(self, output_dir: Path, bucket_name: str, upload_folder: str):
        """Upload all files from output directory except config.yaml"""
        logger.info(f"📤 Starting upload of all files from {output_dir} (excluding config.yaml)")
        
        uploaded_count = 0
        failed_count = 0
        skipped_count = 0
        
        # Walk through all files in output directory recursively
        for file_path in output_dir.rglob('*'):
            if file_path.is_file():
                # Skip config.yaml files
                if file_path.name == 'config.yaml':
                    skipped_count += 1
                    logger.info(f"⏭️ SKIPPED: {file_path.name} (config file)")
                    continue
                    
                try:
                    # Calculate relative path from output directory
                    relative_path = file_path.relative_to(output_dir)
                    remote_path = f"{upload_folder}/{relative_path}"
                    
                    # Convert Windows paths to forward slashes for Supabase
                    remote_path = remote_path.replace('\\', '/')
                    
                    if self.upload_file_to_supabase(file_path, bucket_name, remote_path):
                        uploaded_count += 1
                        logger.info(f"✅ UPLOADED: {relative_path}")
                    else:
                        failed_count += 1
                        logger.warning(f"⚠️ FAILED: {relative_path}")
                        
                except Exception as e:
                    failed_count += 1
                    logger.error(f"❌ ERROR uploading {file_path}: {str(e)}")
        
        logger.info(f"📊 Upload complete: {uploaded_count} successful, {failed_count} failed, {skipped_count} skipped")
        return uploaded_count, failed_count

    def run_training(self, config_content: str, dataset_config: Dict[str, Any], 
                    upload_config: Dict[str, Any]) -> Dict[str, Any]:
        session_id = None
        
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
                    
                    # Upload all files from output directory (except config.yaml)
                    logger.info("📤 Uploading all output files...")
                    uploaded_count, failed_count = self.upload_output_folder(
                        output_dir,
                        upload_config["bucket_name"],
                        upload_config["folder_path"]
                    )
                    
                    return {
                        "success": True,
                        "message": "Training completed successfully",
                        "session_id": session_id,
                        "model_name": model_name,
                        "upload_folder": upload_config["folder_path"],
                        "output_path": str(output_dir),
                        "files_uploaded": uploaded_count,
                        "files_failed": failed_count
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
                    
        except Exception as e:
            logger.error(f"❌ Error in training process: {str(e)}")
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
