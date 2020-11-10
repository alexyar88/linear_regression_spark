package org.apache.spark.ml.made

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.made.WithSparkAndTestData._sqlc
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}

trait WithSparkAndTestData {
  lazy val spark: SparkSession = WithSparkAndTestData._spark
  lazy val sqlc: SQLContext = WithSparkAndTestData._sqlc

  lazy val schema: StructType = new StructType()
    .add("x", DoubleType)
    .add("y", DoubleType)
    .add("z", DoubleType)
    .add("label", DoubleType)

  lazy val df_raw: DataFrame = _sqlc.read
    .option("header", "true")
    .schema(schema)
    .csv("data.csv")

  lazy val assembler: VectorAssembler = new VectorAssembler()
    .setInputCols(Array("x", "y", "z"))
    .setOutputCol("features")

  lazy val df: DataFrame = assembler
    .transform(df_raw)
    .drop("x", "y", "z")
}

object WithSparkAndTestData {
  lazy val _spark: SparkSession = SparkSession.builder
    .appName("Simple Application")
    .master("local[4]")
    .getOrCreate()

  lazy val _sqlc: SQLContext = _spark.sqlContext
}