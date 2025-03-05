# Spark DL Inference on Databricks AWS/Azure

**Note**: fields in \<brackets\> require user inputs.  
Make sure you are in [this](./) directory.

## Setup

1. Install the latest [databricks-cli](https://docs.databricks.com/en/dev-tools/cli/tutorial.html) and configure for your workspace.

2. Specify the path to your Databricks workspace:
    ```shell
    export WS_PATH=</Users/someone@example.com>
    ```

    ```shell
    export SPARK_DL_WS=${WS_PATH}/spark-dl
    databricks workspace mkdirs ${SPARK_DL_WS}
    ```
3. Specify the local paths to the notebook you wish to run, the utils file, and the init script.
    As an example for a PyTorch notebook:
    ```shell
    export NOTEBOOK_SRC=</path/to/notebook_torch.ipynb>
    ```
    ```shell
    export UTILS_SRC=$(realpath ../pytriton_utils.py)
    export INIT_SRC=$(pwd)/setup/init_spark_dl.sh
    ```
4. Specify the framework to torch or tf, corresponding to the notebook you wish to run. Continuing with the PyTorch example:
    ```shell
    export FRAMEWORK=torch
    ```
    This will tell the init script which libraries to install on the cluster.

5. Copy the files to the Databricks Workspace:
    ```shell
    databricks workspace import ${SPARK_DL_WS}/notebook_torch.ipynb --format JUPYTER --file $NOTEBOOK_SRC
    databricks workspace import ${SPARK_DL_WS}/pytriton_utils.py --format AUTO --file $UTILS_SRC
    databricks workspace import ${SPARK_DL_WS}/init_spark_dl.sh --format AUTO --file $INIT_SRC
    ```

6. Launch the cluster with the provided script with the argument `aws` or `azure` based on your provider. 
    ```shell
    cd setup
    chmod +x start_cluster.sh
    ./start_cluster.sh aws  # or ./start_cluster.sh azure
    ```
    By default, the cluster startup script will use the following instances:
    - **Azure torch/tf**: 2x `Standard_NV36ads_A10_v5` workers (1 A10 GPU), 1x `Standard_NV36ads_A10_v5` driver (1 A10 GPU).
    - **Azure vllm**: 2x `Standard_NV72ads_A10_v5` workers (2 A10 GPUs), 1x `Standard_NV36ads_A10_v5` driver (1 A10 GPU).
    - **AWS torch/tf**: 2x `g5.4xlarge` (1 A10 GPU), 1x `g5.2xlarge` driver (1 A10 GPU).
    - **AWS vllm**: 2x `g5.12xlarge` workers (4 A10 GPUs), 1x `g5.2xlarge` driver (1 A10 GPU).
The vllm example requires multiple GPUs per node to demo tensor parallelism. Modify the script if you do not have these specific instance types. 

7. Navigate to the notebook in your workspace and attach it to the cluster. The default cluster name is `spark-dl-inference-$FRAMEWORK`.  