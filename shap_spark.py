from synapse.ml.explainers import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

vec_access = udf(lambda v, i: float(v[i]), FloatType())


def get_spark_session(app_name="adult_SHAP"):
    spark = SparkSession.builder.appName(app_name) \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.parquet.compression.writeLegacyFormat", "true") \
        .getOrCreate()
    return spark


def read_data():
    df = spark.read.option("quotes", "\"").csv("resources/inputs/adult.csv", inferSchema=True, header=True)
    return df


def train():
    labelIndexer = StringIndexer(
        inputCol="income", outputCol="label", stringOrderType="alphabetAsc"
    ).fit(df)
    print("Label index assigment: " + str(set(zip(labelIndexer.labels, [0, 1]))))

    training = labelIndexer.transform(df).cache()
    training.show()
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    categorical_features_idx = [col + "_idx" for col in categorical_features]
    categorical_features_enc = [col + "_enc" for col in categorical_features]
    numeric_features = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    strIndexer = StringIndexer(
        inputCols=categorical_features, outputCols=categorical_features_idx
    )
    onehotEnc = OneHotEncoder(
        inputCols=categorical_features_idx, outputCols=categorical_features_enc
    )
    vectAssem = VectorAssembler(
        inputCols=categorical_features_enc + numeric_features, outputCol="features"
    )
    lr = LogisticRegression(featuresCol="features", labelCol="label", weightCol="fnlwgt")
    pipeline = Pipeline(stages=[strIndexer, onehotEnc, vectAssem, lr])
    model = pipeline.fit(training)
    return model,training,categorical_features,numeric_features


def get_shap_explanations(training,categorical_features,numeric_features):
    explain_instances = (
        model.transform(training).orderBy(lit(1)).limit(5).repartition(200).cache()
    )
    explain_instances.show()

    shap = TabularSHAP(
        inputCols=categorical_features + numeric_features,
        outputCol="shapValues",
        numSamples=5000,
        model=model,
        targetCol="probability",
        targetClasses=[1],
        backgroundData=training.orderBy(lit(1)).limit(100).cache(),
    )

    shap_df = shap.transform(explain_instances)

    shaps = (
        shap_df.withColumn("probability", vec_access(col("probability"), lit(1)))
        .withColumn("shapValues", vector_to_array(col("shapValues").getItem(0)))
        .select(
            ["shapValues", "probability", "label"] + categorical_features + numeric_features
        )
    )

    shaps_local = shaps.toPandas()
    shaps_local.sort_values("probability", ascending=False, inplace=True, ignore_index=True)
    return shaps_local


def create_plot():
    features = categorical_features + numeric_features
    features_with_base = ["Base"] + features

    rows = shaps_local.shape[0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        subplot_titles="Probability: "
                       + shaps_local["probability"].apply("{:.2%}".format)
                       + "; Label: "
                       + shaps_local["label"].astype(str),
    )

    for index, row in shaps_local.iterrows():
        feature_values = [0] + [row[feature] for feature in features]
        shap_values = row["shapValues"]
        list_of_tuples = list(zip(features_with_base, feature_values, shap_values))
        shap_pdf = pd.DataFrame(list_of_tuples, columns=["name", "value", "shap"])
        fig.add_trace(
            go.Bar(
                x=shap_pdf["name"],
                y=shap_pdf["shap"],
                hovertext="value: " + shap_pdf["value"].astype(str),
            ),
            row=index + 1,
            col=1,
        )

    fig.update_yaxes(range=[-1, 1], fixedrange=True, zerolinecolor="black")
    fig.update_xaxes(type="category", tickangle=45, fixedrange=True)
    fig.update_layout(height=400 * rows, title_text="SHAP explanations")
    fig.show()


if __name__ == "__main__":
    spark = get_spark_session()

    try:

        df = read_data()
        model,training,categorical_features,numeric_features = train()
        shaps_local = get_shap_explanations(training,categorical_features,numeric_features)
        create_plot()
    except Exception as err:
        raise err
    finally:
        spark.stop()
