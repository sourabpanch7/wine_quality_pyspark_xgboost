from src.dataAccessObjects.interface import GenericDAO


class CSVDAO(GenericDAO):
    def __init__(self, spark):
        self.spark = spark
        super().__init__(spark=self.spark)

    def read_data(self, **kwargs):
        return self.spark.read.csv(kwargs["path"], inferSchema=True, header=True)

    def write_data(self, **kwargs):
        kwargs["write_data"].coalesce(1).write.mode(kwargs["write_mode"]).csv(kwargs["path"], header=True)


class JSONDAO(GenericDAO):
    def __init__(self, spark):
        self.spark = spark
        super().__init__(spark=self.spark)

    def read_data(self, **kwargs):
        return self.spark.read.option('multiLine', 'true').option('mode', 'PERMISSIVE').json(kwargs["path"])

    def write_data(self, **kwargs):
        kwargs["write_data"].coalsece(1).mode(kwargs["write_mode"]).json(kwargs["path"], multiline=True)


class ParquetDAO(GenericDAO):
    def __init__(self, spark):
        self.spark = spark
        super().__init__(spark=self.spark)

    def read_data(self, **kwargs):
        return self.spark.read.parquet(kwargs["path"], inferSchema=True)

    def write_data(self, **kwargs):
        kwargs["write_data"].coalsece(1).mode(kwargs["write_mode"]).parquet(kwargs["path"])
