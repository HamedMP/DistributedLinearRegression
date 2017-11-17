package se.kth.spark.lab1.task6


import breeze.linalg
import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import breeze.linalg._
import breeze.numerics._


object VectorHelper {
  def dot(v1: Transpose[DenseVector[Double]], v2: DenseVector[Double]): Double = {
    v1 * v2
  }

  def dot(v: DenseVector[Double], s: Double): DenseVector[Double] = {
    v * s
  }

  def sum(v1: DenseVector[Double], v2: DenseVector[Double]): DenseVector[Double] = {
    v1 + v2
  }

  def fill(size: Int, fillVal: Double): DenseVector[Double] = {
    DenseVector.fill(size){fillVal}
  }
}