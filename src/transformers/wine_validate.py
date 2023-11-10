from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from src.transformers.wine_transform import WineTransformer


class WineValidator(WineTransformer):

    @staticmethod
    def validate(self, **kwargs):
        evaluator = MulticlassClassificationEvaluator(
            labelCol="predicted_label", predictionCol="prediction", metricName=kwargs["validation_metric"])
        accuracy = evaluator.evaluate(kwargs["df"])
        return accuracy
