import logging
from src.utils.utilities import read_config, get_spark_session
from src.transformers.wine_train import WineTrain
from src.transformers.wine_validate import WineValidator
from src.transformers.wine_predict import WinePredictor


class WineTrainDriver(WineTrain, WineValidator, WinePredictor):
    @staticmethod
    def run(config_file):
        config_dict = read_config(conf_json=config_file)
        spark = get_spark_session(app_name=config_dict.get("app_name_train", "test"))
        try:
            train_obj = WineTrain(spark=spark)
            df = train_obj.read_data(path=config_dict["ip_path"])
            df = train_obj.transform_data(df=df,label_col=config_dict["label_col"])
            train_df, test_df = train_obj.train_test_split(train_obj, df=df, label_col=config_dict["label_col"],
                                                           train_ratio=config_dict["train_ratio"], seed=10)
            model = train_obj.train_model(train_obj, train_df=train_df, label_col=config_dict["label_col"])
            prediction_obj = WinePredictor(spark=spark)
            pred_df = prediction_obj.predictions(prediction_obj, model=model, df=test_df,
                                                 feature_names=[x.name for x in df.schema if
                                                                x.name != config_dict["label_col"]],
                                                 label_col=config_dict["label_col"])
            pred_df.groupBy(config_dict["label_col"]).count().show()
            validator_obj = WineValidator(spark=spark)
            acc = validator_obj.validate(validator_obj, df=pred_df, label_col=config_dict["label_col"],
                                         validation_metric=config_dict["validation_metric"])
            logging.info(config_dict["validation_metric"] + " is ======> " + str(acc))
            model.save(config_dict["model_path"])

        except Exception as err_msg:
            logging.error(str(err_msg))
            raise err_msg

        finally:
            spark.stop()
