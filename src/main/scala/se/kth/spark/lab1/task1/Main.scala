package se.kth.spark.lab1.task1

import se.kth.spark.lab1._
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.sql.functions._

case class Song(year: Double, f1: Double, f2: Double, f3: Double)

object Main {
  def main(args: Array[String]) {
    val sc = SparkSession.builder.appName("lab1").master("local").getOrCreate()

    import sc.implicits._

    val filePath = "src/main/resources/millionsong.txt"

    val rdd = sc.sparkContext.textFile(filePath)

    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(s => s.split(","))

    //Step3: map each row into a Song object by using the year label and the first three features  
    val songsRdd = recordsRdd.map(s => Song(s(0).toDouble, s(1).toDouble, s(2).toDouble, s(3).toDouble))

    //Step4: convert your rdd into a dataframe
    val songsDf = songsRdd.toDF()

    // Q1
    println(songsDf.count())
    // Q2
    val yearCol = col("year")
    songsDf.groupBy("year").count().filter(yearCol <= 2000 && yearCol >= 1998)
      .agg(sum(col("count")))
      .show()

    // Q3
    songsDf.select(min(yearCol)).show()
    songsDf.select(max(yearCol)).show()
    songsDf.select(avg(yearCol)).show()

    // Q4
    songsDf.groupBy("year").count().filter(yearCol <= 2010 && yearCol >= 2000).sort(col("year")).show()

  }
}