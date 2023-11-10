import logging
import argparse
from src.drivers.wine_train_driver import WineTrainDriver
from src.drivers.wine_predict_driver import WinePredictDriver

if __name__ == '__main__':
    logging.getLogger().setLevel(level=logging.INFO)
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_file", help="config file path")
        parser.add_argument("--job_name", help="job name",default="WineInferenceDriver")
        args = parser.parse_args()
        job_obj = eval(args.job_name)
        job_obj.run(config_file=args.config_file)

    except Exception as err_msg:
        logging.error(str(err_msg))
        raise err_msg
