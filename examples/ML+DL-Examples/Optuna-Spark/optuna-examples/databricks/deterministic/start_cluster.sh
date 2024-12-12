#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

if [[ -z ${INIT_PATH} ]]; then
    echo "Please export INIT_PATH per README.md"
    exit 1
fi

json_config=$(cat <<EOF
{
    "cluster_name": "optuna-xgboost-gpu",
    "spark_version": "15.4.x-gpu-ml-scala2.12",
    "spark_conf": {
        "spark.task.resource.gpu.amount": "1",
        "spark.executor.cores": "8",
        "spark.executor.resource.gpu.amount": "1",
        "spark.task.maxFailures": "1"
    },
    "node_type_id": "Standard_NC8as_T4_v3",
    "driver_node_type_id": "Standard_NC8as_T4_v3",
    "spark_env_vars": {
        "LIBCUDF_CUFILE_POLICY": "OFF"
    },
    "autotermination_minutes": 60,
    "enable_elastic_disk": true,
    "init_scripts": [
        {
            "workspace": {
                "destination": "${INIT_PATH}"
            }
        }
    ],
    "runtime_engine": "STANDARD",
    "num_workers": 4
}
EOF
)

databricks clusters create --json "$json_config"