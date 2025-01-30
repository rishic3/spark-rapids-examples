# Spark DL Inference on AWS EMR

## Setup

**Note**: fields in \<brackets\> require user inputs.
Make sure you are in [this](./) directory. 

#### Setup AWS CLI

1. Install the [AWS CLI](https://docs.aws.amazon.com/emr/latest/EMR-on-EKS-DevelopmentGuide/setting-up-cli.html). 

2. Initialize the CLI via `aws configure`. You may need to create access keys by following [Authenticating using IAM user credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-authentication-user.html). 
    You can find your default region name (e.g. United States (Ohio)) on the right of the top navigation bar. Clicking the region name will show the region code (e.g. us-east-2). 
    ```shell
    aws configure
    AWS Access Key ID [None]: <your_access_key>
    AWS Secret Access Key [None]: <your_secret_access_key>
    Default region name [None]: <region-code>
    Default output format [None]: json
    ```

#### Setup EMR Workspace

If you don't already have one, create a new Studio and set it up with VPC/subnet/internet gateway.

3. Create a new Studio:
    - Navigate to the [AWS EMR Console](https://console.aws.amazon.com/emr/) > "Studios" > "Create Studio".
    - Create a new Studio for interactive workloads.

4. Create a VPC, Subnet, Internet Gateway, and Route Table:
    - Create a new VPC in the [AWS VPC Console](https://console.aws.amazon.com/vpc/) > "Create VPC".
    - Create a subnet under "Subnets" > "Create subnet" and select the newly created VPC. For the availability zone, select your region code.
    - Create an Internet gateway under "Internet gateways" > "Create internet gateways". 
        - Once created, attach it to your VPC under "Actions" > "Attach to VPC".
    - Create a Route table under "Route tables" > "Create route table" and attach it to your VPC.
        - Go to "Edit routes" > "Add route".
        - Create a route with destination "0.0.0.0/0" and target "Internet Gateway", selecting your newly created internet gateway.

5. Associate your Studio with the VPC and subnet you created:
    - Click on the Studio you created and click "Edit". 
    - Under "Networking and Security" > "VPC", select your VPC.
    - Under "Networking and Security" > "Subnets", select your subnet.

6. Create a Workspace in your Studio:
    - Navigate to your Studio dashboard under "Studio Access URL".
    - Name and create a Workspace, e.g. "spark-dl-inference". 

#### Copy Files to S3

7. Create an S3 bucket if you don't already have one.
    ```shell
    export S3_BUCKET=<your_s3_bucket_name>
    aws s3 mb s3://${S3_BUCKET}
    ```

8. Upload the initialization script to S3.
    ```shell
    export INIT_PATH=s3://${S3_BUCKET}/init/init-spark-dl.sh
    aws s3 cp setup/init-spark-dl.sh $INIT_PATH
    ```

#### Create Cluster

9. Export the SubnetId for the Subnet that contains your Studio (created in step 4).
    ```shell
    export SUBNET_ID=<your_SubnetId>
    ```

10. Specify the framework for the cluster, e.g., for a notebook ending in `_torch`:
    ```shell
    export FRAMEWORK=torch
    ```
11. Create the cluster. This command below will create 4 (effective) GPU nodes.
    Note that EMR cherry picks one node (either CORE or TASK) to run JupyterLab service for notebooks and will not use the node for compute.
    ```shell
    export CUR_DIR=$(pwd)

    aws emr create-cluster \
    --name spark_dl_${FRAMEWORK} \
    --release-label emr-7.3.0 \
    --ebs-root-volume-size=32 \
    --applications Name=Hadoop Name=Livy Name=Spark Name=JupyterEnterpriseGateway \
    --service-role EMR_DefaultRole \
    --log-uri s3://${S3_BUCKET}/logs \
    --ec2-attributes SubnetId=${SUBNET_ID},InstanceProfile=EMR_EC2_DefaultRole \
    --instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=g4dn.2xlarge \
                    InstanceGroupType=CORE,InstanceCount=5,InstanceType=g4dn.2xlarge \
    --configurations file://${CUR_DIR}/setup/init-configurations.json \
    --bootstrap-actions Name='Spark DL Bootstrap Action',Path=${INIT_PATH},Args=["${FRAMEWORK}"]
    ```

#### Attach and Run Notebook

11. In the [AWS EMR console](https://console.aws.amazon.com/emr/) > "Clusters", you can find the ClusterId of the created cluster. Wait until all the instances have the Status turned to "Running".

12. Navigate to the Workspace created in Step 6. On the left-side bar, click "Compute" > "EMR on EC2 cluster". 

13. Select the newly created cluster and click "Attach".

13. Upload the Spark DL notebook you wish to run and select the **PySpark kernel**. The notebook is ready to run!
