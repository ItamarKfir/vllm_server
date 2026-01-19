import subprocess
import sys
import os
import psutil
import signal
import atexit

def kill_vllm_processes():
    """Kill all vLLM-related processes"""
    try:
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('vllm' in str(arg).lower() or 'EngineCore' in str(arg) for arg in cmdline):
                    pid = proc.info['pid']
                    if pid != current_pid:
                        print(f"🛑 Killing vLLM process: PID {pid}")
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception as e:
        print(f"⚠️  Error killing processes: {e}")

# Global reference to the subprocess
server_process = None

def cleanup():
    """Cleanup function to kill all child processes"""
    global server_process
    if server_process:
        print("\n🛑 Shutting down vLLM server and child processes...")
        try:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
        except:
            pass
    
    kill_vllm_processes()
    print("✅ Cleanup complete")

def signal_handler(signum, frame):
    """Handle termination signals"""
    print(f"\n⚠️  Received signal {signum}, cleaning up...")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup)

print("=" * 60)
print("Starting vLLM server...")
print("💡 Use 'nvidia-smi' to monitor GPU usage")
print("💡 Press Ctrl+C to stop the server and clean up all processes")
print("=" * 60)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTHONHASHSEED'] = '0'

# Configuration
GPU_UTIL = 0.90
GPU_KV_CACHE_GB = 3.0
GPU_KV_CACHE_BYTES = int(GPU_KV_CACHE_GB * 1024**3)
MAX_MODEL_LEN = 12000
MAX_NUM_SEQS = 8
MAX_NUM_BATCHED_TOKENS = 2048

print("\nSERVER CONFIGURATION:")
print(f"  GPU Memory Utilization: {GPU_UTIL} ({GPU_UTIL*100:.0f}%)")
print(f"  GPU KV Cache: {GPU_KV_CACHE_GB} GiB")
print(f"  Max Model Length: {MAX_MODEL_LEN} tokens")
print(f"  Max Num Seqs: {MAX_NUM_SEQS}")
print(f"  Max Batched Tokens: {MAX_NUM_BATCHED_TOKENS}")
print("=" * 60)
print("Starting vLLM server...\n")

if __name__ == "__main__":
    try:
        server_process = subprocess.Popen([
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", "Qwen2.5-32B-Instruct-AWQ",
            "--port", "8000",
            "--quantization", "awq_marlin",
            "--kv-cache-memory-bytes", str(GPU_KV_CACHE_BYTES),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--max-num-seqs", str(MAX_NUM_SEQS),
            "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
        ])
        
        server_process.wait()
        
    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt received")
        cleanup()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        cleanup()