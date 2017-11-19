package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.Row

import scala.math._

case class Instance(label: Double, features: Vector)

object Helper {
  def rmse(labelsAndPreds: RDD[(Double, Double)]): Double = {
    val n: Double = labelsAndPreds.count()
    val normalizedSum = labelsAndPreds.map(x => pow(x._1 - x._2, 2) / n)
      .reduce(_ + _)

    sqrt(normalizedSum)
  }

  def predictOne(weights: Vector, features: Vector): Double = {
    VectorHelper.dot(weights, features)
  }

  def predict(weights: Vector, data: RDD[Instance]): RDD[(Double, Double)] = {
    data.map(i => (i.label, predictOne(weights, i.features)))
  }
}

class MyLinearRegressionImpl(override val uid: String)
    extends MyLinearRegression[Vector, MyLinearRegressionImpl, MyLinearModelImpl] {

  def this() = this(Identifiable.randomUID("mylReg"))

  override def copy(extra: ParamMap): MyLinearRegressionImpl = defaultCopy(extra)

  def gradientSummand(weights: Vector, lp: Instance): Vector = {
    VectorHelper.dot(lp.features,
      VectorHelper.dot(weights, lp.features) - lp.label) // TODO
  }

  def gradient(d: RDD[Instance], weights: Vector): Vector = {
    d.map(x => gradientSummand(weights, x))
      .reduce((x, y) => VectorHelper.sum(x, y))
  }

  def linregGradientDescent(trainData: RDD[Instance], numIters: Int):
  (Vector, Array[Double]) = {

    val n = trainData.count()
    val d = trainData.take(1)(0).features.size
    var weights = VectorHelper.fill(d, 0)
    val alpha = 0.1
    val errorTrain = Array.fill[Double](numIters)(0.0)

    for (i <- 0 until numIters) {
      //compute this iterations set of predictions based on our current weights
      val labelsAndPredsTrain = Helper.predict(weights, trainData)
      //compute this iteration's RMSE
      errorTrain(i) = Helper.rmse(labelsAndPredsTrain)

      //compute gradient
      val g = gradient(trainData, weights)
      //update the gradient step - the alpha
      val alpha_i = alpha / (n * scala.math.sqrt(i + 1))
      val wAux = VectorHelper.dot(g, (-1) * alpha_i)
      //update weights based on gradient
      weights = VectorHelper.sum(weights, wAux)
    }
    (weights, errorTrain)
  }

  def train(dataset: Dataset[_]): MyLinearModelImpl = {
    println("Training")

    val numIters = 10

    val instances: RDD[Instance] = dataset.select(
      col($(labelCol)), col($(featuresCol))).rdd.map {
        case Row(label: Double, features: Vector) =>
          Instance(label, features)
      }

    val (weights, trainingError) = linregGradientDescent(instances, numIters)
    new MyLinearModelImpl(uid, weights, trainingError)
  }
}

class MyLinearModelImpl(override val uid: String, val weights: Vector, val trainingError: Array[Double])
    extends MyLinearModel[Vector, MyLinearModelImpl] {

  override def copy(extra: ParamMap): MyLinearModelImpl = defaultCopy(extra)

  def predict(features: Vector): Double = {
    println("Predicting")
    val prediction = Helper.predictOne(weights, features)
    prediction
  }
}