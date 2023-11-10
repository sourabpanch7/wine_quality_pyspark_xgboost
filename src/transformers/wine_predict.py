from xgboost.spark import SparkXGBClassifierModel
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array
from src.transformers.wine_transform import WineTransformer


class WinePredictor(WineTransformer):

    @staticmethod
    def predictions(self, **kwargs):
        predict_df = kwargs["model"].transform(kwargs["df"])
        return predict_df.withColumn("predictions_array", vector_to_array("probability")) \
            .withColumn("predicted_label", F.expr("array_position(predictions_array,array_max(predictions_array))")) \
            .withColumn(kwargs["label_col"], F.when(F.col("predicted_label") == F.lit(0), F.lit("Poor"))
                        .when(F.col("predicted_label") == F.lit(1), F.lit("Average"))
                        .otherwise(F.lit("Good"))) \
            .select(kwargs["feature_names"] + [kwargs["label_col"], "prediction","predicted_label"])

    @staticmethod
    def load_model(self, **kwargs):
        return SparkXGBClassifierModel.load(kwargs["model_path"])
