package org.apache.spark.ml.made

import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.DataFrame

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSparkAndTestData {

  private def validateModel(model: LinearRegressionModel, data: DataFrame): Unit = {
    val df_result = model.transform(df)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("mse")

    val mse = evaluator.evaluate(df_result)
    mse should be < mseLimit
  }

  val paramsDelta = 0.01
  val mseLimit = 0.00001

  "Estimator" should "learn params" in {
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setLearningRate(1.0)
      .setNumIters(100)

    val model = lr.fit(df)
    val params = model.getWeights()

    params(0) should be(0.1 +- paramsDelta)
    params(1) should be(0.2 +- paramsDelta)
    params(2) should be(0.3 +- paramsDelta)
    params(3) should be(0.4 +- paramsDelta)
  }

  "Model" should "predict test data with low MSE" in {
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setLearningRate(1.0)
      .setNumIters(100)

    val model = lr.fit(df)
    validateModel(model, df)
  }

  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setLearningRate(1.0)
        .setNumIters(100)
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val model = Pipeline
      .load(tmpFolder.getAbsolutePath)
      .fit(df)
      .stages(0)
      .asInstanceOf[LinearRegressionModel]

    validateModel(model, df)
  }


  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setLearningRate(1.0)
        .setNumIters(100)
    ))

    val model = pipeline.fit(df)
    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModel(reRead.stages(0).asInstanceOf[LinearRegressionModel], df)
  }
}
