import json
from functools import lru_cache
from pyspark.sql import SparkSession

@lru_cache()
def get_spark_session(app_name):
    return SparkSession.builder.appName(app_name) \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.parquet.compression.writeLegacyFormat", "true") \
        .getOrCreate()


def read_config(conf_json):
    with open(conf_json, "r") as cnf_file:
        cnf = json.load(cnf_file)

    return cnf
