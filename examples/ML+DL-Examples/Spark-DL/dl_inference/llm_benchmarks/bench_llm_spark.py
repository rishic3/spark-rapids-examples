import os
import socket
import pandas as pd
import numpy as np
import time
import signal
from functools import partial
from pyspark.sql.types import StringType
from pyspark import SparkConf
from pyspark import TaskContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, struct, length, lit, concat
from pyspark.ml.functions import predict_batch_udf
from gpu_monitor import GPUMonitor
from pytriton_utils import TritonServerManager

# plugin:
# export RAPIDS_JAR=/home/rishic/spark-rapids-examples/examples/ML+DL-Examples/Spark-DL/dl_inference/llm_benchmarks/rapids-4-spark_2.12-24.12.1.jar
# spark-submit --properties-file spark-llm-rapids-config.conf --jars "rapids-4-spark_2.12-24.12.1.jar" bench_llm_spark.py
# no plugin:
# spark-submit --properties-file spark-llm-config.conf bench_llm_spark.py

# triton benchmark:
# spark-submit --properties-file spark-llm-config-16-core.conf bench_llm_spark.py

def triton_server(ports):
    import time
    import signal
    import numpy as np
    from pytriton.decorators import batch
    from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
    from pytriton.triton import Triton, TritonConfig
    from pyspark import TaskContext
    from vllm import LLM, SamplingParams

    print(f"SERVER: Initializing model on worker {TaskContext.get().partitionId()}.")
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=128)
    llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", dtype="bfloat16", max_num_seqs=512, max_num_batched_tokens=400000)

    @batch
    def _infer_fn(**inputs):
        prompts = np.squeeze(inputs["prompts"]).tolist()
        decoded_prompts = [p.decode("utf-8") for p in prompts]
        outputs = llm.generate(decoded_prompts, sampling_params)
        return {
            "outputs": np.array([o.outputs[0].text for o in outputs]).reshape(-1, 1)
        }

    workspace_path = f"/tmp/triton_{TaskContext.get().partitionId()}_{time.strftime('%m_%d_%M_%S')}"
    triton_conf = TritonConfig(http_port=ports[0], grpc_port=ports[1], metrics_port=ports[2])
    with Triton(config=triton_conf, workspace=workspace_path) as triton:
        triton.bind(
            model_name="qwen-2.5",
            infer_func=_infer_fn,
            inputs=[
                Tensor(name="prompts", dtype=object, shape=(-1,)),
            ],
            outputs=[
                Tensor(name="outputs", dtype=object, shape=(-1,)),
            ],
            config=ModelConfig(
                max_batch_size=512,
                batcher=DynamicBatcher(max_queue_delay_microseconds=50000),  # 50ms
            ),
            strict=True,
        )

        def _stop_triton(signum, frame):
            print("SERVER: Received SIGTERM. Stopping Triton server.")
            triton.stop()

        signal.signal(signal.SIGTERM, _stop_triton)

        print("SERVER: Serving inference")
        triton.serve()

def triton_fn(model_name, host_to_url):
    from pytriton.client import ModelClient
    import socket


    url = host_to_url.get(socket.gethostname())
    client = ModelClient(url, model_name, inference_timeout_s=500)
    print(f"Connecting to Triton model {model_name} at {url}.")

    def infer_batch(inputs):
        flattened = np.squeeze(inputs).tolist()
        # Encode batch
        encoded_batch = [[text.encode("utf-8")] for text in flattened]
        encoded_batch_np = np.array(encoded_batch, dtype=np.bytes_)
        # Run inference
        result_data = client.infer_batch(encoded_batch_np)
        result_data = np.squeeze(result_data["outputs"], -1)
        return result_data
        
    return infer_batch

def main():
    spark = SparkSession.builder.appName("spark-llm-plugin").getOrCreate()
    sc = spark.sparkContext
    sc.addPyFile("pytriton_utils.py")

    os.environ["SPARK_HOME"] = "/opt/spark-3.5.3"
    
    if spark.conf.get("spark.plugins", None) is None:
        log_prefix = "no_plugin"
    else:
        log_prefix = "with_plugin"
    
    # Start Triton Servers
    model_name = "qwen-2.5"
    num_nodes = 1
    server_manager = TritonServerManager(num_nodes, model_name)

    start_server_time = time.perf_counter()

    server_manager.start_servers(triton_server, wait_timeout=24)
    host_to_http_url = server_manager.host_to_http_url

    file_path = "/home/rishic/spark-rapids-examples/examples/ML+DL-Examples/Spark-DL/dl_inference/llm_benchmarks/spark-dl-datasets/pubmed_abstracts_5k.parquet"

    # Define system prompt and predict function
    system_prompt = '''You are a knowledgeable AI assistant. Your job is to create a 2-3 sentence summary 
of a research abstract that captures the main objective, methodology, and key findings, using clear and concise
language while preserving technical accuracy and quantitative results.'''

    generate = predict_batch_udf(partial(triton_fn, model_name=model_name, host_to_url=host_to_http_url),
                                return_type=StringType(),
                                input_tensor_shapes=[[1]],
                                batch_size=256)
    
    # Start monitoring
    monitor = GPUMonitor()
    monitor.start()

    try:
        start_read = time.perf_counter()

        # Read and preprocess data
        df = spark.read.parquet(file_path)
        num_parts = df.rdd.getNumPartitions()
        print("Num partitions", num_parts)
        df = df.select(
            concat(
                lit("<|im_start|>system\n"),
                lit(system_prompt),
                lit("<|im_end|>\n<|im_start|>user\n"),
                col("value"),
                lit("<|im_end|>\n<|im_start|>assistant\n")
            ).alias("prompt")
        )
        preds = df.withColumn("response", generate(col("prompt")))
        preds.write.mode("overwrite").parquet("spark-dl-datasets/pubmed_abstracts_5k_preds.parquet")

        end_write = time.perf_counter()

        # Logging
        logtime = monitor.log_file.split(".")[-2].split("metrics")[1]
        print(f"E2E read -> inference -> write time: {end_write - start_read:.4f} seconds")
        print(f"Triton server start time: {start_read - start_server_time:.4f} seconds")
        parent_dir = "/home/rishic/spark-rapids-examples/examples/ML+DL-Examples/Spark-DL/dl_inference/llm_benchmarks/results"
        with open(f"{parent_dir}/out_times_{log_prefix}{logtime}.txt", "w") as f:
            f.write(f"Start read time: {start_read}\nEnd write time: {end_write}\nNum partitions: {num_parts}\n")
    finally:
        monitor.stop()
        server_manager.stop_servers()

if __name__ == "__main__":
    main()