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
        
        # Auto-discover the correct training script path
        self.toolkit_script = self._find_training_script()

    def _find_training_script(self):
        """Find the correct path to the AI toolkit training script"""
        possible_paths = [
            "/app/ai-toolkit/toolkit/train.py",
            "/app/ai-toolkit/train.py", 
            "/app/toolkit/train.py",
            "/usr/local/bin/train.py"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found training script at: {path}")
                return path
        
        # If not found, try alternative approach
        logger.info("Training script not found, trying directory-based approach")
        return "/app/ai-toolkit"

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
            session_id = f"training_{int(time.time())}"
            session_dir = self.training_dir / session_id
            session_dir.mkdir(exist_ok=True)
            logger.info(f"Created training session: {session_id}")

            dataset_local_path = session_dir / "dataset"
            self.download_dataset(
                dataset_config['bucket_name'],
                dataset_config['folder_path'],
                dataset_local_path
            )

            config = yaml.safe_load(config_content)
            if 'config' in config and 'process' in config['config']:
                for process_item in config['config']['process']:
                    if 'datasets' in process_item:
                        for dataset in process_item['datasets']:
                            if 'folder_path' in dataset:
                                dataset['folder_path'] = str(dataset_local_path)

            output_dir = session_dir / "outputs"
            output_dir.mkdir(exist_ok=True)

            if 'config' in config and 'process' in config['config']:
                for process_item in config['config']['process']:
                    if 'save' in process_item:
                        process_item['save']['save_dir'] = str(output_dir)
                    else:
                        process_item['save'] = {'save_dir': str(output_dir)}

            config_file = session_dir / "training_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            logger.info(f"Starting training with config: {config_file}")
            logger.info(f"Dataset path: {dataset_local_path}")
            logger.info(f"Output directory: {output_dir}")

            monitor_thread = threading.Thread(
                target=self.monitor_and_upload_outputs,
                args=(output_dir, upload_config['bucket_name'], upload_config['folder_path']),
                daemon=True
            )
            monitor_thread.start()

            # Improved command construction with multiple fallback approaches
            if self.toolkit_script.endswith('.py'):
                cmd = ["python", self.toolkit_script, "--config", str(config_file)]
            else:
                # Try running from the toolkit directory
                cmd = ["python", "-m", "train", "--config", str(config_file)]
                # Change to the toolkit directory
                os.chdir(self.toolkit_script)

            logger.info(f"Executing command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

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

            logger.info("Training completed, uploading final files...")
            time.sleep(10)

            self.upload_file_to_supabase(
                config_file,
                upload_config['bucket_name'],
                f"{upload_config['folder_path']}/config/training_config.yaml"
            )

            result = {
                "success": return_code == 0,
                "return_code": return_code,
                "session_id": session_id,
                "output_lines": output_lines[-50:],
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
