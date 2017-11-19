package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.{DenseVector, Matrices, Vector, Vectors}


object VectorHelper {

  def dot(v1: Vector, v2: Vector): Double = {
    (v1.toArray, v2.toArray).zipped.map((x, y) => x * y).sum
  }

  def dot(v: Vector, s: Double): Vector = {
    new DenseVector(v.toArray.map(_ * s))
  }

  def sum(v1: Vector, v2: Vector): Vector = {
    new DenseVector((v1.toArray, v2.toArray).zipped.map(_ + _))
  }

  def fill(size: Int, fillVal: Double): Vector = {
    new DenseVector(Array.fill(size){fillVal})
  }
}