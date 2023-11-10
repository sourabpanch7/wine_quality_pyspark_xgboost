spark-submit \
--verbose \
--master local[*] \
--archives $3 \
--py-files dist/wine_quality_classification-0.1.0-py3.9.egg \
main.py \
--config_file $2 \
--job_name $1

#--conf "spark.pyspark.python=/Users/sourabpanchanan/anaconda3/envs/wine_quality_pyspark_xgboost/bin/python" \
