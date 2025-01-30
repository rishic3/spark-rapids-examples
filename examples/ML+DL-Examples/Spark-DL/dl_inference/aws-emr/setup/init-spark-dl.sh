#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.
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
set -euxo pipefail

FRAMEWORK=$1
echo "Framework: $FRAMEWORK"

# Set up cgroup for RAPIDS
sudo mkdir -p /spark-rapids-cgroup/devices
sudo mount -t cgroup -o devices cgroupv1-devices /spark-rapids-cgroup/devices
sudo chmod a+rwx -R /spark-rapids-cgroup

sudo chown -R $USER:$USER /home/

# install conda
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_PATH="$HOME/miniconda3"

if [ ! -d "$MINICONDA_PATH" ]; then
    wget "$MINICONDA_URL" -O miniconda.sh
    sudo bash miniconda.sh -b -p "$MINICONDA_PATH"
    rm miniconda.sh
fi

export PATH="$MINICONDA_PATH/bin:$PATH"
eval "$($MINICONDA_PATH/bin/conda shell.bash hook)"
$MINICONDA_PATH/bin/conda init bash

which conda
# create environment
conda create -y -n spark-dl-${FRAMEWORK} python=3.11
conda activate spark-dl-${FRAMEWORK}

COMMON_REQUIREMENTS="numpy
pandas
matplotlib
portalocker
pyarrow
h5py
pydot
scikit-learn
huggingface
datasets==3.*
transformers
urllib3<2"

if [[ "${FRAMEWORK}" == "torch" ]]; then
    requirements="${COMMON_REQUIREMENTS}
torch==2.5.1
torchvision
torch-tensorrt
tensorrt --extra-index-url https://download.pytorch.org/whl/cu121
sentence_transformers
sentencepiece
nvidia-modelopt[all] --extra-index-url https://pypi.nvidia.com"

elif [[ "${FRAMEWORK}" == "tf" ]]; then
    requirements="${COMMON_REQUIREMENTS}
tensorflow[and-cuda]
tf-keras"
else
    echo "Unexpected framework argument: expected torch or tf"
    exit 1
fi

cat <<EOF > temp_requirements.txt
${requirements}
EOF

which pip3
pip3 install --upgrade pip
pip3 --version

pip3 install --no-cache-dir -r temp_requirements.txt
rm temp_requirements.txt

echo "Installing Docker..."
sudo yum update -y
sudo yum install -y docker git
sudo service docker start
sudo usermod -a -G docker hadoop

git clone https://github.com/triton-inference-server/pytriton.git
cd pytriton

# comment this out to avoid pip uninstall error
# sudo make install-dev
echo "PYTHON PATH:"
which python3
pip3 install build
pip3 install --extra-index-url https://pypi.ngc.nvidia.com -e .[dev]
sudo env "PATH=$PATH" "PYTHON=$(which python3)" make dist

ls dist
pip3 install dist/nvidia_pytriton-*-py3-none-*.whl

which python
python -c "import nvidia.pytriton; print(nvidia.pytriton.__version__)"

# hack to prevent livy from overriding driver python used for jupyter kernel
cat <<EOF >/tmp/mod_start_kernel.sh
#!/bin/bash
set -ex
while [ ! -f /mnt/notebook-env/bin/start_kernel_as_emr_notebook.sh ]; do
echo "waiting for /mnt/notebook-env/bin/start_kernel_as_emr_notebook.sh"
sleep 10
done
echo "done waiting"
sleep 10
sudo sed -i /mnt/notebook-env/bin/start_kernel_as_emr_notebook.sh -e 's#"spark.pyspark.python": "python3"#"spark.pyspark.python": "/home/hadoop/.conda/envs/spark-dl-torch/bin/python"#g'
sudo sed -i /mnt/notebook-env/bin/start_kernel_as_emr_notebook.sh -e 's#"spark.pyspark.virtualenv.enabled": "true"#"spark.pyspark.virtualenv.enabled": "false"#g'
exit 0
EOF
sudo bash /tmp/mod_start_kernel.sh &
exit 0
