# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ---- TODO: Give credit to @vonodiripsa for the base class!!! ---- #

import logging.config
import torch
import logging
from sentence_transformers import SentenceTransformer
from pyspark.ml.functions import predict_batch_udf
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params
from pyspark.sql.types import (
    ArrayType,
    FloatType,
)


class HuggingFaceSentenceEmbedder(Transformer, HasInputCol, HasOutputCol):
    """
    Custom transformer that extends PySpark's Transformer class to
    perform sentence embedding using a model with optional TensorRT acceleration.
    """

    NUM_OPT_ROWS = 100  # Constant for number of rows taken for model optimization
    BATCH_SIZE_DEFAULT = 64

    runtime = Param(
        Params._dummy(),
        "runtime",
        "Specifies the runtime environment: cpu, cuda, or tensorrt",
    )
    batchSize = Param(Params._dummy(), "batchSize", "Batch size for embeddings", int)
    modelName = Param(Params._dummy(), "modelName", "Full Model Name parameter", str)

    def __init__(
        self,
        inputCol=None,
        outputCol=None,
        runtime=None,
        batchSize=None,
        modelName=None,
    ):
        """
        Initialize the HuggingFaceSentenceEmbedder with input/output columns and optional TRT flag.
        """
        super(HuggingFaceSentenceEmbedder, self).__init__()

        self.logger = logging.getLogger("embedder")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

        # Determine the default runtime based on CUDA availability
        default_runtime = "cuda" if torch.cuda.is_available() else "cpu"

        # Override the provided runtime if CUDA is not available
        effective_runtime = runtime if torch.cuda.is_available() else "cpu"

        self._setDefault(
            runtime=default_runtime,
            batchSize=self.BATCH_SIZE_DEFAULT,
        )
        self._set(
            inputCol=inputCol,
            outputCol=outputCol,
            runtime=effective_runtime,
            batchSize=batchSize if batchSize is not None else self.BATCH_SIZE_DEFAULT,
            modelName=modelName,
        )
        self.optData = None
        self.model = None
        self.row_count = 0  # This should be set when the DataFrame is available
        self.predict_batch_udf = None  # Placeholder for predict_batch_udf, cached on first pass
        
        # These variables will be set if startTriton is called:
        self.serve_with_triton = False
        self.pids = {}
        self.num_nodes = 0

    def setInputCol(self, value):
        self._set(inputCol=value)
        return self
    
    def getInputCol(self):
        return self.getOrDefault(self.inputCol)
    
    def setOutputCol(self, value):
        self._set(outputCol=value)
        return self
    
    def getOutputCol(self):
        return self.getOrDefault(self.outputCol)

    def setBatchSize(self, value):
        self._set(batchSize=value)
        return self

    def getBatchSize(self):
        return self.getOrDefault(self.batchSize)

    def setRuntime(self, value):
        if value not in ["cpu", "cuda", "tensorrt"]:
            raise ValueError(
                "Invalid runtime specified. Choose from 'cpu', 'cuda', 'tensorrt'"
            )
        self.setOrDefault(self.runtime, value)

    def getRuntime(self):
        return self.getOrDefault(self.runtime)

    def setModelName(self, value):
        self._set(modelName=value)
        return self

    def getModelName(self):
        return self.getOrDefault(self.modelName)

    def setRowCount(self, row_count):
        self.row_count = row_count
        # Override the runtime if row count is less than 100 or CUDA is not available
        if self.row_count < 100 or not torch.cuda.is_available():
            self.set(self.runtime, "cpu")
        return self

    # ----- Triton Methods ----- #

    def _triton_server(self):
        """
        Initialize and serve the model using Triton server.
        """
        from pytriton.decorators import batch
        from pytriton.triton import Triton
        from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
        import model_navigator as nav
        import signal
        import numpy as np
        
        runtime = self.getRuntime()
        assert runtime in ("tensorrt", "cuda"), "Triton inference requires a GPU runtime."

        server_logger = logging.getLogger("triton_server")
        server_logger.info(f"SERVER: Initializing '{self.getModelName()}' model with runtime: {runtime}.")

        if self.model == None:
            self.model = self._init_model()
        
        @batch
        def _infer_func(**inputs):
            sentences = np.squeeze(inputs["text"]).tolist()
            server_logger.info(f"SERVER: Received batch of {len(sentences)} sentences.")
            decoded_sentences = [s.decode("utf-8") for s in sentences]
            return self.model.encode(decoded_sentences, convert_to_tensor=False, show_progress_bar=False)
        
        with Triton() as triton:
            triton.bind(
                model_name=self.getModelName(),
                infer_func=_infer_func,
                inputs=[
                    Tensor(name="text", dtype=object, shape=(-1,)),
                ],
                outputs=[
                    Tensor(name="embeddings", dtype=object, shape=(-1,)),
                ],
                config=ModelConfig(
                    max_batch_size=self.BATCH_SIZE_DEFAULT * 2,
                    batcher=DynamicBatcher(max_queue_delay_microseconds=5000),
                )
            )

            def _stop_triton(signum, frame):
                server_logger.info("Received SIGTERM. Stopping Triton server.")
                triton.stop()

            signal.signal(signal.SIGTERM, _stop_triton)

            server_logger.info(f"SERVER: Serving inference.")
            triton.serve()

    def _start_triton(self):
        """
        Launch the Triton server in a separate process on each node.
        """
        import socket
        from multiprocessing import Process
        from pytriton.client import ModelClient

        hostname = socket.gethostname()
        
        process = Process(target=self._triton_server)
        process.start()
        self.logger.info(f"Starting Triton server process with PID {process.pid}. Waiting for {self.getModelName()} to be ready.")

        client = ModelClient("localhost", self.getModelName())
        ready = False
        while not ready:
            try:
                client.wait_for_server(5)
                ready = True
            except Exception as e:
                self.logger.info(f"Waiting for server to be ready: {e}")
        
        return [(hostname, process.pid)]
    
    def _stop_triton(self):
        """
        Stop the Triton server processes on all nodes.
        """
        import os
        import socket
        import signal
        import time 
        
        hostname = socket.gethostname()
        pid = self.pids.get(hostname, None)
        assert pid is not None, f"Could not find pid for {hostname}"
        os.kill(pid, signal.SIGTERM)
        time.sleep(7)
        
        for _ in range(5):
            try:
                os.kill(pid, 0)
            except OSError:
                return [True]
            time.sleep(3)

        return [False]


    def _use_stage_level_scheduling(self, spark, rdd):
        """
        Request stage-level resources (1 GPU per task) for the RDD.
        """
        if spark.version < "3.4.0":
            raise Exception("Stage-level scheduling is not supported in Spark < 3.4.0")

        executor_cores = spark.conf.get("spark.executor.cores")
        assert executor_cores is not None, "spark.executor.cores is not set"
        executor_gpus = spark.conf.get("spark.executor.resource.gpu.amount")
        assert executor_gpus is not None and int(executor_gpus) <= 1, "spark.executor.resource.gpu.amount must be set and <= 1"

        from pyspark.resource.profile import ResourceProfileBuilder
        from pyspark.resource.requests import TaskResourceRequests

        spark_plugins = spark.conf.get("spark.plugins", " ")
        assert spark_plugins is not None
        spark_rapids_sql_enabled = spark.conf.get("spark.rapids.sql.enabled", "true")
        assert spark_rapids_sql_enabled is not None

        task_cores = (
            int(executor_cores)
            if "com.nvidia.spark.SQLPlugin" in spark_plugins
            and "true" == spark_rapids_sql_enabled.lower()
            else (int(executor_cores) // 2) + 1
        )

        task_gpus = 1.0

        treqs = TaskResourceRequests().cpus(task_cores).resource("gpu", task_gpus)
        rp = ResourceProfileBuilder().require(treqs).build

        self.logger.info(f"Reqesting stage-level resources: (cores={task_cores}, gpu={task_gpus})")

        return rdd.withResources(rp)
    
    def _predict_triton_fn(self):
        """
        Create and return predict function for batch inference using Triton server.
        """
        from pytriton.client import ModelClient
        import numpy as np

        self.logger.info("Connecting to Triton server on localhost.")

        def infer_batch(inputs):

            with ModelClient("localhost", self.getModelName(), init_timeout_s=600) as client:
                flattened = np.squeeze(inputs).tolist()
                encoded_batch = np.array([[s.encode("utf-8") for s in flattened]], dtype=np.bytes_)
                
                self.logger.info(f"Triton predict: {len(flattened)}")
                result_data = client.infer_batch(encoded_batch)
                embeddings = np.squeeze(result_data["embeddings"], -1)
                return embeddings
        
        return infer_batch
    
    # ----- End Triton Methods ----- #

    def _init_model(self):
        """
        Initialize the model and optimize it if necessary.
        """
        runtime = self.getRuntime()
        global model
        modelName = self.getModelName()

        print(f"Initializing model '{modelName}' with runtime '{runtime}'.")

        model = SentenceTransformer(
            modelName, device="cpu" if runtime == "cpu" else "cuda"
        ).eval()

        if runtime in ("tensorrt"):
            import tensorrt as trt
            import model_navigator as nav
            
            # this forces navigator to use specific runtime
            nav.inplace_config.strategy = nav.SelectedRuntimeStrategy(
                "trt-fp16", "TensorRT"
            )

            # Accept either 'sentence-transformers/xxx' or 'xxx'
            try:
                moduleName = modelName.split("/")[1]
            except IndexError:
                moduleName = modelName

            try:
                # Optimized model is already cached.
                nav.load_optimized()
                self.logger.info("Loaded cached model.")
            except Exception:
                # Model hasn't been optimized/cached. Optimize with TensorRT. 
                model = nav.Module(model, name=moduleName, forward_func="forward")
                conf = nav.OptimizeConfig(
                    target_formats=(nav.Format.TENSORRT,),
                    runners=("TensorRT",),
                    optimization_profile=nav.OptimizationProfile(
                        max_batch_size=self.BATCH_SIZE_DEFAULT
                    ),
                    custom_configs=[
                        nav.TorchConfig(autocast=True),
                        nav.TorchScriptConfig(autocast=True),
                        nav.TensorRTConfig(
                            precision=(nav.TensorRTPrecision.FP16,),
                            onnx_parser_flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM.value],
                        ),
                    ],
                )

                def _get_dataloader():
                    input_data = self.optData
                    return [
                        (
                            0,
                            (
                                input_data,
                                {"show_progress_bar": False, "batch_size": self.getBatchSize()},
                            ),
                        )
                    ]
                
                self.logger.info(f"Optimizing model '{modelName}' with TensorRT.")
                nav.optimize(model.encode, dataloader=_get_dataloader(), config=conf)
                nav.load_optimized()

        return model

    def _predict_batch_fn(self):
        """
        Create and return a function for batch prediction.
        """

        if self.model == None:
            self.model = self._init_model()

        def predict(inputs):
            """
            Predict method to encode inputs using the model.
            """
            with torch.no_grad():
                self.logger.info(f"Predict: {len(inputs)}")
                output = self.model.encode(
                    inputs.tolist(), convert_to_tensor=False, show_progress_bar=False
                )

            return output

        return predict

    def _transform(self, input_df):
        """
        Apply the transformation to the input dataframe.
        """
        input_col = self.getInputCol()
        output_col = self.getOutputCol()

        size = input_df.count()
        self.setRowCount(size)
        if size >= self.NUM_OPT_ROWS:
            sample = input_df.take(self.NUM_OPT_ROWS)
            self.optData = [row[input_col] for row in sample]
        else:
            self.logger.warn(f"Number of rows in the input dataframe is less than {self.NUM_OPT_ROWS}. \
                             Using the entire dataframe for optimization.")
            self.optData = [row[input_col] for row in input_df.collect()]

        if self.predict_batch_udf is None:
            if self.serve_with_triton:
                assert self.pids != {}, "Triton server is not running. Call startTriton() to start the server."
                self.predict_batch_udf = predict_batch_udf(
                    self._predict_triton_fn(),
                    return_type=ArrayType(FloatType()),
                    batch_size=self.getBatchSize(),
                )
            else:
                self.predict_batch_udf = predict_batch_udf(
                    self._predict_batch_fn,
                    return_type=ArrayType(FloatType()),
                    batch_size=self.getBatchSize(),
                )

        return input_df.withColumn(output_col, self.predict_batch_udf(input_col))

    def transform(self, input_df):
        """
        Public method to transform the dataframe.
        """
        return self._transform(input_df)
    
    def startTriton(self, num_nodes):
        """
        Public method to start Triton server on all nodes. This will enable inference using Triton via serve_with_triton flag.
        
        Args:
            num_nodes (int): Number of nodes to start Triton server on, = number of nodes in cluster.
        """
        import json
        from pyspark.sql import SparkSession

        assert self.pids == {}, "Triton server is already running."
        self.num_nodes = num_nodes
        
        spark = SparkSession.builder.getOrCreate()
        sc = spark.sparkContext

        nodeRDD = sc.parallelize(list(range(num_nodes)), num_nodes)
        nodeRDD = self._use_stage_level_scheduling(spark, nodeRDD)

        self.logger.info("Starting Triton servers on all nodes.")
        self.pids = nodeRDD.barrier().mapPartitions(lambda _: self._start_triton()).collectAsMap()
        print(f"Triton Server PIDs:\n", json.dumps(self.pids, indent=4))
        self.serve_with_triton = True
    
    def stopTriton(self):
        """
        Public method to stop Triton server on all nodes.
        """
        from pyspark.sql import SparkSession
        
        assert self.pids != {} and self.num_nodes > 0, "Triton server is not running."

        spark = SparkSession.builder.getOrCreate()
        sc = spark.sparkContext

        shutdownRDD = sc.parallelize(list(range(self.num_nodes)), self.num_nodes)
        shutdownRDD = self._use_stage_level_scheduling(spark, shutdownRDD)

        self.logger.info("Stopping Triton servers on all nodes.")
        result = shutdownRDD.barrier().mapPartitions(lambda _: self._stop_triton()).collect()
        self.logger.info(f"Triton Server Stopped: {result}")
        self.serve_with_triton = False
        self.pids = {}
