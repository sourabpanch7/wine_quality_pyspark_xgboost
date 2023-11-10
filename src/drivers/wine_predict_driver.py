import logging
from src.utils.utilities import read_config, get_spark_session
from src.transformers.wine_predict import WinePredictor


class WinePredictDriver(WinePredictor):
    @staticmethod
    def run(config_file):
        config_dict = read_config(conf_json=config_file)
        spark = get_spark_session(app_name=config_dict.get("app_name_predict", "test"))
        try:
            prediction_obj = WinePredictor(spark=spark)
            pred_data = prediction_obj.read_data(path=config_dict["predict_data_path"])
            model = prediction_obj.load_model(prediction_obj, model_path=config_dict["model_path"])
            pred_df = prediction_obj.predictions(prediction_obj, model=model, df=pred_data,
                                                 label_col=config_dict["label_col"],
                                                 feature_names=[x.name for x in pred_data.schema if
                                                                x.name != config_dict["label_col"]])
            pred_df = pred_df.select(pred_data.columns + [config_dict["label_col"]])
            pred_df.groupBy(config_dict["label_col"]).count().show()

            prediction_obj.write_data(path=config_dict["inference_op_path"], write_data=pred_df, write_mode="overwrite")
        except Exception as err_msg:
            logging.error(str(err_msg))
            raise err_msg

        finally:
            spark.stop()
