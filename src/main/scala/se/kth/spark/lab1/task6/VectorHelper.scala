package se.kth.spark.lab1.task6


import breeze.linalg
import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import breeze.linalg._
import breeze.numerics._


object VectorHelper {
  def dot(v1: Vector, v2: Vector): Double = {
    linalg.Vector(v1) * linalg.Vector(v2)
  }

  def dot(v: Vector, s: Double): Vector = {
    linalg.Vector(v) *:* s
  }

  def sum(v1: Vector, v2: Vector): Vector = {
    linalg.Vector(v1) + linalg.Vector(v2)
  }

  def fill(size: Int, fillVal: Double): DenseVector[Double] = {
    DenseVector.fill(size){fillVal}
  }
}