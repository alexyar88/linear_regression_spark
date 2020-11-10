package org.apache.spark.ml.made

import breeze.linalg.{DenseVector => BVector, sum}
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.regression.{RegressionModel, Regressor}
import org.apache.spark.ml.param.{ParamMap, DoubleParam, IntParam}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, MetadataUtils}
import org.apache.spark.ml.PredictorParams
import org.apache.spark.sql.{Dataset, Row}


trait LinearRegressionParams extends PredictorParams {

  final val learningRate: DoubleParam = new DoubleParam(this, "learningRate", "learning rate")
  final val numIters: IntParam = new IntParam(this, "numIters", "number of iters")

  setDefault(learningRate, 1.0)
  setDefault(numIters, 100)

  def setLearningRate(value: Double): this.type = set(learningRate, value)

  def setNumIters(value: Int): this.type = set(numIters, value)
}


class LinearRegression(override val uid: String)
  extends Regressor[Vector, LinearRegression, LinearRegressionModel] with LinearRegressionParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def copy(extra: ParamMap): LinearRegression = ???

  override protected def train(dataset: Dataset[_]): LinearRegressionModel = {
    val numFeatures = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
    var weights: BVector[Double] = BVector.zeros(numFeatures + 1)
    val gradCol = "grad"

    val transformUdf = dataset.sqlContext.udf.register(uid + "_grad",
      (x_no_ones: Vector, y: Double) => {
        val one = BVector(1.0)
        val x = BVector.vertcat(one, x_no_ones.asBreeze.toDenseVector)
        val grad = x * (sum(x * weights) - y)
        Vectors.fromBreeze(grad)
      }
    )

    for (_ <- 0 to $(numIters)) {
      val dataset_transformed = dataset.withColumn(gradCol, transformUdf(dataset($(featuresCol)), dataset($(labelCol))))
      val Row(Row(grad_mean_arr)) = dataset_transformed
        .select(Summarizer.metrics("mean").summary(dataset_transformed(gradCol)))
        .first()

      val grad_mean: BVector[Double] = grad_mean_arr.asInstanceOf[DenseVector].asBreeze.toDenseVector
      weights = weights - $(learningRate) * grad_mean
    }
    val params = Vectors.fromBreeze(weights)

    copyValues(new LinearRegressionModel(params)).setParent(this)
  }
}

class LinearRegressionModel protected[made](override val uid: String, weights: Vector)
  extends RegressionModel[Vector, LinearRegressionModel] with PredictorParams {

  def this(weights: Vector) = this(Identifiable.randomUID("linearRegressionModel"), weights)

  override def predict(features: Vector): Double = {
    val one = BVector(1.0)
    val x = BVector.vertcat(one, features.asBreeze.toDenseVector)
    sum(x * weights.asBreeze.toDenseVector)
  }

  override def copy(extra: ParamMap): LinearRegressionModel = ???

  def getWeights(): BVector[Double] = {
    weights.asBreeze.toDenseVector
  }
}
