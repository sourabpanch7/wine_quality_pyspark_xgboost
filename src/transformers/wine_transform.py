from pyspark.sql import functions as F
from src.dataAccessObjects.DAO import CSVDAO


class WineTransformer(CSVDAO):

    def transform_data(self, **kwargs):
        return kwargs["df"].withColumn(kwargs["label_col"], F.when(F.col("quality") <= F.lit(3.3), F.lit(0))
                                       .when(F.col("quality") >= F.lit(6.6), F.lit(2))
                                       .otherwise(F.lit(1))
                                       ).drop("quality")

