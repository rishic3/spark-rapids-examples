import os
import ray
import time
import logging

from ray import serve
from gpu_monitor import GPUMonitor
from functools import partial
import pandas as pd

from vllm import LLM, SamplingParams

class VLLMPredictor:
    def __init__(self):
        print(f"Initializing model on GPU {ray.get_gpu_ids()[0]}")
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            max_tokens=256,
        )
        self.llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16")

    def __call__(self, input_batch):
        outputs = self.llm.generate(input_batch["value"].tolist(), self.sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        input_batch["response"] = responses
        return input_batch

def preprocess_batch(batch: pd.DataFrame, system_prompt: str) -> pd.DataFrame:
    """Preprocessing function for Ray Data pipeline"""
    batch["value"] = batch["value"].apply(lambda text: (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    ))
    return batch

def log_timing_info(start_read, end_write, gpu_monitor):
    """Log timing information for benchmarking"""
    logtime = gpu_monitor.log_file.split(".")[-2].split("metrics")[1]
    parent_dir = "/home/rishic/spark-rapids-examples/examples/ML+DL-Examples/Spark-DL/dl_inference/llm_benchmarks/results"
    with open(f"{parent_dir}/out_times_ray{logtime}.txt", "w") as f:
        f.write(f"Start read time: {start_read}\n"
                f"End write time: {end_write}\n")
    return

def main():
    ray.init(
        runtime_env={
            "conda": "/home/rishic/anaconda3/envs/spark-dl-vllm",
            "env_vars": {
                "LD_LIBRARY_PATH": "/home/rishic/anaconda3/envs/spark-dl-vllm/lib:/rishic/anaconda3/envs/spark-dl-vllm/lib/python3.11/site-packages/nvidia_pytriton.libs:$LD_LIBRARY_PATH"
                }
        }
    )

    file_path = "/home/rishic/spark-rapids-examples/examples/ML+DL-Examples/Spark-DL/dl_inference/llm_benchmarks/spark-dl-datasets/pubmed_abstracts_5k.parquet"

    system_prompt = """You are a knowledgeable AI assistant. Your job is to create a 2-3 sentence summary 
    of a research abstract that captures the main objective, methodology, and key findings, using clear 
    language while preserving technical accuracy and quantitative results."""

    # Start monitoring GPU utilization
    monitor = GPUMonitor()
    monitor.start()

    try:
        start_read = time.perf_counter()

        # Begin Ray job: read parquet -> preprocess -> inference -> write parquet
        ds = ray.data.read_parquet(file_path)
        ds = ds.map_batches(partial(preprocess_batch, system_prompt=system_prompt),
                            num_cpus=1,
                            concurrency=8,
                            batch_format="pandas")
        ds = ds.map_batches(VLLMPredictor,
                            num_gpus=1,
                            concurrency=1,
                            batch_size=64)
        ds.write_parquet("spark-dl-datasets/pubmed_abstracts_5k_ray_preds.parquet")

        end_write = time.perf_counter()
        
        # Logging
        logtime = monitor.log_file.split(".")[-2].split("metrics")[1]
        print(f"E2E read -> inference -> write time: {end_write - start_read:.4f} seconds")
        parent_dir = "/home/rishic/spark-rapids-examples/examples/ML+DL-Examples/Spark-DL/dl_inference/llm_benchmarks/results"
        with open(f"{parent_dir}/out_times_ray{logtime}.txt", "w") as f:
            f.write(f"Start read time: {start_read}\nEnd write time: {end_write}\n")
    finally:
        monitor.stop()
        serve.shutdown()

if __name__ == "__main__":
    main()