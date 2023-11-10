/Users/sourabpanchanan/work/spark/spark-3.4.2-bin-hadoop3/bin/spark-submit \
--verbose \
--master local[*] \
--packages com.microsoft.azure:synapseml_2.12:1.0.2 \
--conf "spark.pyspark.python=/Users/sourabpanchanan/anaconda3/envs/wine_quality_pyspark_xgboost/bin/python" \
shap_spark.py
