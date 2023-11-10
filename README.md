# Pyspark XGBoost Multiclass Classification

This is a sample PySpark Xgboost classification project. It contains train and inference jobs.

## Initial assumptions

1. This code is to be run on an existing spark distribution where spark >= 3.2.
2. Please ensure that the python version in the environment where the .egg file is created and where it's to be
   executed,
   match.
3. Anaconda is installed in all nodes of the cluster.The below steps are for setting up the environment using Anaconda.
   Please refer to the below link for more details:

https://docs.anaconda.com/anaconda/install/index.html

## Steps to Run the code

#### Create egg file

1. Ensure that you are in the wine_quality_pyspark_xgboost directory.
2. Create the egg file using the below command.

**<"PATH TO DESIRED PYTHON INSTALLATION"> setup.py bdist_egg**

#### Package conda environment to run via spark-submit

1. Activate the necessary conda environment using the below command.

   "conda activate **<env_name>**"

2. Package the environment using the below command.

   "conda pack -f -o **<env_path>**"

## Steps to Launch the App

1. Run the app via the **_spark_submit.sh_** script by providing the necessary command line arguments.

For e.g..

"sh spark_submit.sh WinePredictDriver resources/configs/wine_classification_config.json
envs/wine_quality_pyspark_xgboost.tar.gz"

Here
$1 => Job Name (Should match the driver class name)
$2 => Config file name
$3 => Packaged env path

## Steps to add new Jobs

0. Ensure that the DAO object for reading and writing from your desired source is present in _**src.dataAccessObjects**_
1. Create corresponding transformer,train,validate and predictions classes under **_src.transformers_**
2. Create corresponding driver class under **_src.drivers_**