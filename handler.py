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
        
        # Comprehensive diagnostics
        self.toolkit_info = self._comprehensive_toolkit_discovery()

    def _comprehensive_toolkit_discovery(self):
        """Comprehensive AI toolkit discovery"""
        logger.info("=== COMPREHENSIVE TOOLKIT DISCOVERY ===")
        
        info = {
            "found_scripts": [],
            "found_directories": [],
            "python_modules": [],
            "executable_files": []
        }
        
        # 1. Search for Python scripts
        possible_script_paths = [
            "/app/ai-toolkit/toolkit/train.py",
            "/app/ai-toolkit/train.py", 
            "/app/toolkit/train.py",
            "/app/ai-toolkit/scripts/train.py",
            "/app/train.py",
            "/usr/local/bin/train.py",
            "/opt/ai-toolkit/train.py"
        ]
        
        for path in possible_script_paths:
            if os.path.exists(path):
                info["found_scripts"].append(path)
                logger.info(f"✅ Found script: {path}")
        
        # 2. Search for directories
        possible_dirs = [
            "/app/ai-toolkit",
            "/app/toolkit", 
            "/opt/ai-toolkit",
            "/usr/local/ai-toolkit"
        ]
        
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                info["found_directories"].append(dir_path)
                logger.info(f"✅ Found directory: {dir_path}")
                
                # List contents
                try:
                    contents = list(Path(dir_path).rglob("*.py"))[:10]  # First 10 .py files
                    logger.info(f"   Python files: {[str(f) for f in contents]}")
                except Exception as e:
                    logger.info(f"   Error listing contents: {e}")
        
        # 3. Try to find train.py anywhere
        logger.info("Searching for train.py files...")
        try:
            result = subprocess.run(
                ["find", "/", "-name", "train.py", "-type", "f", "2>/dev/null"], 
                capture_output=True, text=True, timeout=10
            )
            train_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            info["found_scripts"].extend([f for f in train_files if f not in info["found_scripts"]])
            logger.info(f"✅ Found train.py files: {train_files}")
        except Exception as e:
            logger.info(f"❌ Find command failed: {e}")
        
        # 4. Check if toolkit can be imported as a module
        try:
            import sys
            sys.path.append('/app/ai-toolkit')
            import toolkit
            info["python_modules"].append("toolkit (via /app/ai-toolkit)")
            logger.info("✅ Can import toolkit module")
        except Exception as e:
            logger.info(f"❌ Cannot import toolkit: {e}")
        
        # 5. Check for executable files
        executable_paths = [
            "/app/ai-toolkit/run_training.sh",
            "/app/ai-toolkit/toolkit.py",
            "/usr/local/bin/ai-toolkit",
            "/usr/local/bin/toolkit"
        ]
        
        for path in executable_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                info["executable_files"].append(path)
                logger.info(f"✅ Found executable: {path}")
        
        # 6. Check what's in /app/ai-toolkit specifically
        if os.path.exists("/app/ai-toolkit"):
            logger.info("=== /app/ai-toolkit directory structure ===")
            try:
                for root, dirs, files in os.walk("/app/ai-toolkit"):
                    level = root.replace("/app/ai-toolkit", "").count(os.sep)
                    indent = " " * 2 * level
                    logger.info(f"{indent}{os.path.basename(root)}/")
                    subindent = " " * 2 * (level + 1)
                    for file in files[:5]:  # First 5 files per directory
                        logger.info(f"{subindent}{file}")
                    if len(files) > 5:
                        logger.info(f"{subindent}... and {len(files) - 5} more files")
            except Exception as e:
                logger.info(f"Error exploring directory: {e}")
        
        logger.info(f"=== DISCOVERY SUMMARY ===")
        logger.info(f"Scripts found: {info['found_scripts']}")
        logger.info(f"Directories found: {info['found_directories']}")
        logger.info(f"Python modules: {info['python_modules']}")
        logger.info(f"Executables: {info['executable_files']}")
        
        return info

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
                samples_dir = output_dir / "samples"
                if samples_dir.exists():
                    for sample_file in samples_dir.glob("**/*"):
                        if sample_file.is_file() and sample_file not in uploaded_files:
                            relative_path = sample_file.relative_to(output_dir)
                            remote_path = f"{upload_folder}/outputs/{relative_path}"
                            if self.upload_file_to_supabase(sample_file, bucket_name, remote_path):
                                uploaded_files.add(sample_file)
                for checkpoint in output_dir.glob("*.safetensors"):
                    if checkpoint not in uploaded_files:
                        remote_path = f"{upload_folder}/checkpoints/{checkpoint.name}"
                        if self.upload_file_to_supabase(checkpoint, bucket_name, remote_path):
                            uploaded_files.add(checkpoint)
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
            # First, return the diagnostic information
            return {
                "success": False,
                "diagnostic_info": self.toolkit_info,
                "message": "This is a diagnostic run. Check the logs for detailed discovery information.",
                "next_steps": "Based on the discovered files, we'll create the correct training command."
            }
            
        except Exception as e:
            logger.error(f"Error in diagnostics: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "diagnostic_info": self.toolkit_info
            }

def handler(event):
    input_data = event.get("input", {})
    logger.info(f"Received training request: {json.dumps(input_data, indent=2)}")

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
    logger.info("Starting RunPod serverless AI-Toolkit handler...")
    runpod.serverless.start({"handler": handler})
