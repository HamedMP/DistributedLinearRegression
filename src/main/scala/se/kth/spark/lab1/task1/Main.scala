package se.kth.spark.lab1.task1

import se.kth.spark.lab1._

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

case class Song(yaer: Double, f1: Double, f2: Double, f3: Double)

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._
    
    val filePath = "src/main/resources/millionsong.txt"

    val rdd = sc.textFile(filePath)
    val rawDF = rdd.toDF("raw").cache()
    //Step1: print the first 5 rows, what is the delimiter, number of features and the data types?
    rawDF.take(5).foreach(println)

    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(s=>s.split(","))
    
    //Step3: map each row into a Song object by using the year label and the first three features  
    val songsRdd = recordsRdd.map(s => Song(s(0).toDouble,s(1).toDouble,s(2).toDouble,s(3).toDouble))

    //Step4: convert your rdd into a datafram
    val songsDf = songsRdd.toDF()
//    songsDf.collect().foreach(println)
  }
}