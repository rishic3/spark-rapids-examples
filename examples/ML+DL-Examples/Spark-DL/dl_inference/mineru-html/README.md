# Spark + MinerU-HTML

## Installation

### 1. Create Conda Env
```shell
conda create -n spark-miner python=3.11 -y
conda activate spark-miner
```

### 2. Install MinerU-HTML
```shell
git clone https://github.com/opendatalab/MinerU-HTML.git
cd MinerU-HTML
pip install .
```

### 3. Install Additional Dependencies
```shell
# cd to this directory
cd .. && cd spark-rapids-examples/examples/ML+DL-Examples/Spark-DL/dl_inference/mineru-html
pip install -r requirements.txt
```

## Start Spark Cluster

Start a local Standalone cluster with a single GPU executor. 

*Note:* Make sure your PySpark version (`conda list pyspark`) matches your Spark installation version.
```shell
# Replace with your Spark installation path
export SPARK_HOME=</path/to/spark>
```

```shell
# Configure and start cluster
export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=1
export CORES_PER_WORKER=8
export SPARK_WORKER_OPTS="-Dspark.worker.resource.gpu.amount=1 \
                          -Dspark.worker.resource.gpu.discoveryScript=$SPARK_HOME/examples/src/main/scripts/getGpusResources.sh"
${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-worker.sh -c ${CORES_PER_WORKER} -m 16G ${MASTER}
```
