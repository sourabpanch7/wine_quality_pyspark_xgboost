from xgboost.spark import SparkXGBClassifier
from src.transformers.wine_transform import WineTransformer


class WineTrain(WineTransformer):

    @staticmethod
    def train_test_split(self, **kwargs):
        distinct_values = kwargs["df"].select(kwargs["label_col"]).distinct().rdd.map(lambda x: int(x[0])).collect()
        fractions_dict = {k: kwargs["train_ratio"] for k in distinct_values}
        train_df = kwargs["df"].sampleBy(kwargs["label_col"], fractions=fractions_dict, seed=kwargs["seed"])
        test_df = kwargs["df"].subtract(train_df)
        return train_df, test_df

    @staticmethod
    def train_model(self, **kwargs):
        label_name = kwargs["label_col"]
        feature_names = [x.name for x in kwargs["train_df"].schema if x.name != label_name]
        classifier = SparkXGBClassifier(
            features_col=feature_names,
            label_col=label_name,
            num_workers=2,
            device="cuda",
            missing=0.0
        )

        model = classifier.fit(kwargs["train_df"])
        return model
