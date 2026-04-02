#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch inference script for multiple models
Reads inference task configuration from config file, executes benchmark_test.py using each model's corresponding environment
"""

import os
import sys
import json
import subprocess
import argparse
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List


# Root directory
BASE_DIR = Path(__file__).parent.absolute()
MODELS_DIR = BASE_DIR / 'models'
ENVS_DIR = BASE_DIR / 'envs'


def load_config(config_path: Path) -> Dict:
    """Load configuration file"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Validate configuration
    if 'dataset' not in config:
        raise ValueError("Missing 'dataset' field in config file")
    if 'models' not in config:
        raise ValueError("Missing 'models' field in config file")
    
    return config


def get_python_path(env_name: str) -> str:
    """Get Python path for specified environment"""
    env_path = ENVS_DIR / env_name
    
    # Try python
    python_path = env_path / 'bin' / 'python'
    if python_path.exists():
        return str(python_path)
    
    # Try python3
    python3_path = env_path / 'bin' / 'python3'
    if python3_path.exists():
        return str(python3_path)
    
    raise FileNotFoundError(f"Python interpreter not found for environment {env_name}: {env_path / 'bin'}")


def run_model_inference(
    model_config: Dict,
    dataset_path: str
) -> tuple:
    """
    Run inference for a single model
    
    Args:
        model_config: Model configuration dict containing name, script_path, env_name, device, extra_args
        dataset_path: Dataset path
    
    Returns:
        (success: bool, message: str, duration: float)
    """
    model_name = model_config['name']
    script_path = MODELS_DIR / model_name / model_config['script_path']
    env_name = model_config['env_name']
    device = model_config.get('device', 'cuda')
    extra_args = model_config.get('extra_args', {})
    
    # Check if script exists
    if not script_path.exists():
        return False, f"Script not found: {script_path}", 0.0
    
    # Get Python path
    try:
        python_path = get_python_path(env_name)
    except FileNotFoundError as e:
        return False, str(e), 0.0
    
    # Build command
    cmd = [
        python_path,
        str(script_path),
        '--dataset', str(dataset_path),
        '--device', device
    ]
    
    # Add extra arguments
    for key, value in extra_args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
            else:
                continue
        elif isinstance(value, (list, tuple)):
            cmd.append(f'--{key}')
            cmd.extend([str(v) for v in value])
        else:
            cmd.append(f'--{key}')
            cmd.append(str(value))
    
    # Record start time
    start_time = time.time()
    
    # Execute command
    try:
        print(f"\n{'='*80}")
        print(f"Starting model: {model_name}")
        print(f"Script path: {script_path}")
        print(f"Python path: {python_path}")
        print(f"Dataset: {dataset_path}")
        print(f"Device: {device}")
        if extra_args:
            print(f"Extra arguments: {extra_args}")
        print(f"{'='*80}\n")
        
        # Set working directory to model directory
        work_dir = MODELS_DIR / model_name
        
        # Execute inference (display output in real-time)
        stdout_lines = []
        stderr_lines = []
        
        # Use Popen to read output in real-time
        process = subprocess.Popen(
            cmd,
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffering
        )
        
        # Read and display output in real-time
        def read_output(pipe, is_stderr=False):
            """Read output and display/record simultaneously (filter tqdm updates)"""
            lines = []
            last_tqdm_line = None
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        # tqdm usually starts with \r for overwriting the same line, or contains progress bar features
                        is_tqdm_update = (
                            line.startswith('\r') or 
                            '\r' in line or
                            ('%' in line and ('it/s' in line or 's/it' in line or 'ETA' in line))
                        )
                        
                        if is_tqdm_update:
                            # tqdm update: display only, don't record (avoid log explosion)
                            # Remove \r and preceding content, show only current progress
                            clean_line = line.replace('\r', '').strip()
                            if clean_line:
                                print(f'\r{clean_line}', end='', flush=True)
                                last_tqdm_line = clean_line
                        else:
                            # Normal output: display and record
                            line = line.rstrip()
                            if line:
                                # If there was tqdm output before, newline first
                                if last_tqdm_line:
                                    print()  # Newline to end tqdm line
                                    last_tqdm_line = None
                                
                                lines.append(line)
                                # Display to console in real-time
                                print(line, flush=True)
            except Exception:
                pass
            finally:
                # Ensure the last tqdm line is also newlined
                if last_tqdm_line:
                    print()
                pipe.close()
            return lines
        
        # Create threads to read stdout and stderr
        stdout_thread = threading.Thread(
            target=lambda: stdout_lines.extend(read_output(process.stdout, False))
        )
        stderr_thread = threading.Thread(
            target=lambda: stderr_lines.extend(read_output(process.stderr, True))
        )
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete (with timeout)
        try:
            return_code = process.wait(timeout=100000)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return_code = -1
        
        # Wait for output threads to complete
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        
        duration = time.time() - start_time
        
        if return_code == 0:
            print(f"\n✓ {model_name} inference completed, duration: {duration:.2f}s")
            return True, "Inference successful", duration
        else:
            error_msg = f"Inference failed, return code: {return_code}"
            if stderr_lines:
                error_msg += f"\nError message: {stderr_lines[-1][:500] if stderr_lines else 'No error message'}"
            print(f"\n✗ {model_name} {error_msg}")
            return False, error_msg, duration
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        error_msg = f"Inference timeout (exceeded 1 hour)"
        print(f"✗ {model_name} {error_msg}")
        return False, error_msg, duration
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Execution exception: {str(e)}"
        print(f"✗ {model_name} {error_msg}")
        return False, error_msg, duration


def main():
    parser = argparse.ArgumentParser(
        description='Batch inference script for multiple models (reads configuration from config file)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python run_all_models_inference.py --config inference_config.json
  python run_all_models_inference.py -c inference_config.json
        """
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='inference_config.json',
        help='Config file path (default: inference_config.json)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config).absolute()
    if not config_path.is_absolute() and not (BASE_DIR / config_path).exists():
        config_path = BASE_DIR / args.config
    
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error: Failed to load config file: {e}")
        sys.exit(1)
    
    # Get dataset path
    dataset_path = Path(config['dataset']).absolute()
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)
    
    # Get execution mode
    execution_mode = config.get('execution_mode', 'sequential').lower()
    if execution_mode not in ['sequential', 'parallel']:
        print(f"Warning: Unknown execution mode '{execution_mode}', using default 'sequential'")
        execution_mode = 'sequential'
    
    # Get result output directory
    result_dir = config.get('log_dir', './logs')
    if result_dir:
        result_dir = Path(result_dir)
        if not result_dir.is_absolute():
            result_dir = BASE_DIR / result_dir
        result_dir.mkdir(parents=True, exist_ok=True)
    else:
        result_dir = BASE_DIR
    
    # Filter enabled models
    enabled_models = [
        model for model in config['models']
        if model.get('enabled', True)
    ]
    
    if not enabled_models:
        print("Error: No enabled models in config file")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Batch Inference Task")
    print(f"Config file: {config_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Execution mode: {execution_mode}")
    print(f"Number of models: {len(enabled_models)}")
    print(f"{'='*80}\n")
    
    # Print model list
    print("Enabled models:")
    for i, model in enumerate(enabled_models, 1):
        device = model.get('device', 'cuda')
        print(f"  {i}. {model['name']:15s} | Device: {device:10s} | Script: {model['script_path']}")
    print()
    
    # Execute inference
    results = []
    total_start_time = time.time()
    
    if execution_mode == 'parallel':
        # Parallel execution (all models run simultaneously, including models on the same GPU)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        print("Running in parallel mode (all models execute simultaneously)...")
        print(f"Will run {len(enabled_models)} models simultaneously")
        
        # Display GPU allocation
        gpu_allocation = {}
        for model_config in enabled_models:
            device = model_config.get('device', 'cuda')
            if device not in gpu_allocation:
                gpu_allocation[device] = []
            gpu_allocation[device].append(model_config['name'])
        
        print("\nGPU allocation:")
        for device, models in gpu_allocation.items():
            print(f"  {device}: {len(models)} models - {', '.join(models)}")
        print()
        
        # Use thread pool to run all models simultaneously
        # Note: Use threads instead of processes, as subprocess already creates independent processes
        results_dict = {}
        results_lock = threading.Lock()
        
        def run_single_model(model_config):
            """Run a single model and return result"""
            model_name = model_config['name']
            device = model_config.get('device', 'cuda')
            
            print(f"[{device}] Starting model: {model_name}")
            success, message, duration = run_model_inference(
                model_config,
                str(dataset_path)
            )
            
            result = (model_name, success, message, duration)
            with results_lock:
                results_dict[model_name] = result
            
            status = "Success" if success else "Failed"
            print(f"[{device}] Completed model: {model_name} ({status}, {duration:.2f}s)")
            
            return result
        
        # Create thread pool to run all models simultaneously
        with ThreadPoolExecutor(max_workers=len(enabled_models)) as executor:
            # Submit all tasks
            future_to_model = {
                executor.submit(run_single_model, model_config): model_config['name']
                for model_config in enabled_models
            }
            
            # Wait for all tasks to complete
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Model {model_name} execution error: {e}")
                    with results_lock:
                        results_dict[model_name] = (model_name, False, f"Execution exception: {str(e)}", 0.0)
        
        # Sort results by original order
        for model_config in enabled_models:
            model_name = model_config['name']
            if model_name in results_dict:
                results.append(results_dict[model_name])
            else:
                results.append((model_name, False, "Not executed", 0.0))
    else:
        # Sequential execution
        print("Running in sequential mode...")
        for model_config in enabled_models:
            success, message, duration = run_model_inference(
                model_config,
                str(dataset_path)
            )
            results.append((model_config['name'], success, message, duration))
    
    total_duration = time.time() - total_start_time
    
    # Summarize results
    print(f"\n{'='*80}")
    print(f"Inference Task Completed")
    print(f"{'='*80}\n")
    
    success_count = sum(1 for r in results if r[1])
    fail_count = len(results) - success_count
    
    print(f"Total: {len(results)} models")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)\n")
    
    print("Detailed results:")
    print("-" * 80)
    for model_name, success, message, duration in results:
        status = "✓ Success" if success else "✗ Failed"
        print(f"{status:8s} | {model_name:15s} | {duration:8.2f}s | {message[:50]}")
    print("-" * 80)
    
    # Save summary to file
    dataset_name = dataset_path.stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = result_dir / f'inference_summary_{dataset_name}_{timestamp}.txt'
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"Batch Inference Task Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"Config file: {config_path}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Execution mode: {execution_mode}\n")
        f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n")
        f.write(f"Total: {len(results)} models\n")
        f.write(f"Success: {success_count}\n")
        f.write(f"Failed: {fail_count}\n")
        f.write(f"Total duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)\n")
        f.write(f"\n")
        f.write(f"Detailed results:\n")
        f.write(f"{'-'*80}\n")
        for model_name, success, message, duration in results:
            status = "✓ Success" if success else "✗ Failed"
            f.write(f"{status:8s} | {model_name:15s} | {duration:8.2f}s | {message}\n")
        f.write(f"{'-'*80}\n")
        f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\nResult summary saved to: {result_file}")
    
    # Return non-zero exit code if there are failed models
    if fail_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
