import runpod
import json
import time

def handler(event):
    input_data = event.get("input", {})
    
    # Validate inputs
    required_fields = ["config", "dataset_config", "upload_config"]
    for field in required_fields:
        if field not in input_data:
            yield {"error": f"Missing required field: {field}"}
            return
    
    try:
        training_handler = TrainingHandler()
        
        # Stream progress updates throughout training
        for progress in training_handler.run_training_with_streaming(
            input_data["config"],
            input_data["dataset_config"],
            input_data["upload_config"]
        ):
            yield progress
            
    except Exception as e:
        yield {"error": str(e)}

class TrainingHandler:
    # ... existing methods ...
    
    def run_training_with_streaming(self, config_content: str, dataset_config: Dict[str, Any], 
                                  upload_config: Dict[str, Any]):
        """Generator function that yields progress updates"""
        
        try:
            # Extract model name and setup
            model_name = extract_model_name_from_config(config_content)
            session_id = f"training_session_{int(time.time())}"
            
            yield {
                "status": "starting",
                "message": f"Starting training for {model_name}",
                "session_id": session_id,
                "model_name": model_name,
                "timestamp": time.time()
            }
            
            # Setup directories
            session_dir = self.training_dir / session_id
            session_dir.mkdir(exist_ok=True)
            
            yield {
                "status": "setup",
                "message": "Setting up training environment",
                "session_dir": str(session_dir)
            }
            
            # Download dataset
            yield {"status": "downloading", "message": "Downloading dataset..."}
            dataset_path = session_dir / "dataset"
            self.download_dataset(dataset_config["bucket_name"], dataset_config["folder_path"], dataset_path)
            
            yield {"status": "downloaded", "message": f"Dataset downloaded to {dataset_path}"}
            
            # Create and update config
            config_file = session_dir / "config.yaml"
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            # Update config with paths
            config_data = yaml.safe_load(config_content)
            # ... config update logic ...
            
            yield {"status": "config_ready", "message": "Configuration updated"}
            
            # Start file monitoring
            output_dir = session_dir / "output"
            output_dir.mkdir(exist_ok=True)
            
            file_observer = self.setup_realtime_file_watcher(
                output_dir, upload_config["bucket_name"], upload_config["folder_path"], model_name
            )
            
            yield {"status": "monitoring", "message": "File monitoring started"}
            
            # Start training process
            cmd = ["python", str(self.run_script), str(config_file)]
            
            yield {"status": "training_started", "message": f"Training command: {' '.join(cmd)}"}
            
            original_cwd = os.getcwd()
            os.chdir(str(self.ai_toolkit_dir))
            
            try:
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, universal_newlines=True
                )
                
                # Stream training output line by line
                for line in iter(process.stdout.readline, ''):
                    if line:
                        line = line.rstrip()
                        yield {
                            "status": "training_output",
                            "message": line,
                            "timestamp": time.time()
                        }
                
                process.wait()
                
                if process.returncode == 0:
                    yield {
                        "status": "completed",
                        "message": "Training completed successfully!",
                        "session_id": session_id,
                        "model_name": model_name,
                        "output_path": str(output_dir)
                    }
                else:
                    yield {
                        "status": "failed",
                        "message": f"Training failed with return code {process.returncode}",
                        "session_id": session_id
                    }
                    
            finally:
                os.chdir(original_cwd)
                if file_observer:
                    file_observer.stop()
                    file_observer.join()
                    
        except Exception as e:
            yield {
                "status": "error",
                "message": str(e),
                "timestamp": time.time()
            }

# Use the streaming handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
