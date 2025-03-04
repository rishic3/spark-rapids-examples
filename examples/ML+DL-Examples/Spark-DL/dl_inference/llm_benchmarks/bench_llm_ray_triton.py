import os
import ray
import time
import signal
import socket
import numpy as np
import pandas as pd
from typing import List, Tuple
from functools import partial
from datetime import datetime
from gpu_monitor import GPUMonitor
import pyarrow.parquet as pq
import pyarrow as pa


@ray.remote(num_gpus=1, num_cpus=6)
class TritonServer:
    def __init__(self, ports: Tuple[int, int, int]):
        self.ports = ports
        self.hostname = socket.gethostname()
        
        # Set GPU based on Ray's resource assignment
        gpu_id = ray.get_gpu_ids()[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
        print(f"SERVER: Initializing model on GPU {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"SERVER: Using HF cache: {os.environ.get('HF_HOME')}")

        # Initialize vLLM
        self.triton = None
        self.llm = None
        self.sampling_params = None
        
        # Start Triton Server
        self._start_server()
        
    def _start_server(self):
        from multiprocessing import Process
        
        def run_server():
            from vllm import LLM, SamplingParams
            from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
            from pytriton.triton import Triton, TritonConfig
            from pytriton.decorators import batch
            
            # Initialize vLLM
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.8,
                repetition_penalty=1.05,
                max_tokens=256
            )
            llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", dtype="float16")

            @batch
            def _infer_fn(**inputs):
                prompts = np.squeeze(inputs["prompts"]).tolist()
                decoded_prompts = [p.decode("utf-8") for p in prompts]
                outputs = llm.generate(decoded_prompts, sampling_params)
                return {
                    "outputs": np.array([o.outputs[0].text for o in outputs]).reshape(-1, 1)
                }

            workspace_path = f"/tmp/triton_ray_{self.ports[0]}_{time.strftime('%m_%d_%M_%S')}"
            triton_conf = TritonConfig(
                http_port=self.ports[0],
                grpc_port=self.ports[1],
                metrics_port=self.ports[2]
            )
            
            with Triton(config=triton_conf, workspace=workspace_path) as triton:
                triton.bind(
                    model_name="qwen-2.5",
                    infer_func=_infer_fn,
                    inputs=[Tensor(name="prompts", dtype=object, shape=(-1,))],
                    outputs=[Tensor(name="outputs", dtype=object, shape=(-1,))],
                    config=ModelConfig(
                        max_batch_size=64,
                        batcher=DynamicBatcher(max_queue_delay_microseconds=5000),
                    ),
                    strict=True,
                )
                triton.serve()
        
        # Start server process
        self.server_process = Process(target=run_server)
        self.server_process.start()
        
        # Wait for server to be ready using ModelClient
        from pytriton.client import ModelClient
        client = ModelClient(f"http://localhost:{self.ports[0]}", "qwen-2.5")
        patience = 30
        for _ in range(patience):
            try:
                client.wait_for_model(5)
                print(f"Server ready on ports {self.ports}")
                return
            except Exception:
                print("Waiting for server to be ready...")
                time.sleep(1)
        
        raise TimeoutError("Server failed to start in time")
        
    def get_connection_info(self):
        return (self.hostname, self.ports)
    
    def shutdown(self):
        if self.server_process:
            self.server_process.terminate()
            self.server_process.join()
        return True

def preprocess_batch(batch: pd.DataFrame, system_prompt: str) -> pd.DataFrame:
    """Preprocessing function similar to Spark's pandas_udf"""
    batch["prompt"] = [
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        for text in batch['text']
    ]
    return batch

def run_inference(batch: pd.DataFrame, server_url: str) -> pd.DataFrame:
    """Run inference on a batch of data"""
    from pytriton.client import ModelClient
    
    with ModelClient(server_url, "qwen-2.5", inference_timeout_s=500) as client:
        encoded_batch = [[text.encode("utf-8")] for text in batch['prompt']]
        encoded_batch_np = np.array(encoded_batch, dtype=np.bytes_)
        result_data = client.infer_batch(encoded_batch_np)
        responses = np.squeeze(result_data["outputs"], -1)
        batch['response'] = responses
        
    return batch

def main():
    ray.init(
        num_cpus=32,
        num_gpus=2,
        runtime_env={
            "conda": "/rishic/anaconda3/envs/spark-dl-vllm",
            "env_vars": {
                "LD_LIBRARY_PATH": "/rishic/anaconda3/envs/spark-dl-vllm/lib:/rishic/anaconda3/envs/spark-dl-vllm/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/rishic/anaconda3/envs/spark-dl-vllm/lib/python3.11/site-packages/nvidia_pytriton.libs:$LD_LIBRARY_PATH",
                "HF_HOME": "/raid/spark-team/rishic/hf_home"
                }
        }
    )

    # Start Triton Servers
    server1 = TritonServer.remote((7000, 7001, 7002))
    server2 = TritonServer.remote((7003, 7004, 7005))
    servers = [server1, server2]
    
    # Get server connection info
    server_info = ray.get([server.get_connection_info.remote() for server in servers])
    print("Server Information:", server_info)

    # Set up round-robin server assignment
    server_urls = [f"grpc://localhost:{info[1][1]}" for info in server_info]
    
    # System prompt
    system_prompt = '''You are a knowledgeable AI assistant. Your job is to create a 2-3 sentence summary 
    of a research abstract that captures the main objective, methodology, and key findings, using clear 
    language while preserving technical accuracy and quantitative results.'''
    
    monitor = GPUMonitor(gpu_ids=[0, 1])
    monitor.start()

    try:
        start_read = time.perf_counter()

        # Read parquet data using Ray Data
        ds = ray.data.read_parquet("/raid/spark-team/rishic/spark-dl-datasets/pubmed_abstracts_10k_40part")
        
        # Preprocess data
        ds = ds.map_batches(
            partial(preprocess_batch, system_prompt=system_prompt),
            batch_size=64,
            concurrency=20,
            num_cpus=1,
        )
        
        # Run inference with server round-robin
        def infer_with_server(batch: pd.DataFrame) -> pd.DataFrame:
            server_id = np.random.randint(0, 2)
            server_url = server_urls[server_id]
            return run_inference(batch, server_url)
        
        ds = ds.map_batches(
            infer_with_server,
            batch_size=64,
            concurrency=20,
            num_cpus=1,
        )
        
        # Write results
        ds.write_parquet("/raid/spark-team/rishic/spark-dl-datasets/qwen_pubmed_results_10k_rayio")
        
        end_write = time.perf_counter()
        
        # Log timing information
        print(f"E2E read -> inference -> write time: {end_write - start_read:.4f} seconds")
        logtime = monitor.log_file.split(".")[-2].split("metrics")[1]
        parent_dir = "/rishic/myforks/spark-rapids-examples/examples/ML+DL-Examples/Spark-DL/dl_inference/llm_bench/results"
        with open(f"{parent_dir}/out_times_ray{logtime}.txt", "w") as f:
            f.write(f"Start read time: {start_read}\n"
                   f"End write time: {end_write}\n")
    
    finally:
        # Stop monitoring
        monitor.stop()
        
        # Shutdown servers
        if 'servers' in locals():
            ray.get([server.shutdown.remote() for server in servers])
        
        # Shutdown Ray
        ray.shutdown()

if __name__ == "__main__":
    main()