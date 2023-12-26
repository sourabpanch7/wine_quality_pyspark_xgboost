<SPARK_HOME>/spark-submit \
--verbose \
--master local[*] \
--packages com.microsoft.azure:synapseml_2.12:1.0.2 \
--conf "spark.pyspark.python=<python_path>" \
shap_spark.py
