package se.kth.spark.lab1.task3

import org.apache.spark._
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.ml.feature.{HashingTF, Tokenizer, RegexTokenizer}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.regression.LinearRegression

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    
    import sqlContext.implicits._
    import sqlContext._
    
    val filePath = "src/main/resources/millionsong.txt"
    val rdd = sc.textFile(filePath)
    val myDF = rdd.map(s=>s.split(","))
    .map(s => (s(0).toDouble, Array(s(1).toDouble,s(2).toDouble,s(3).toDouble)))
    .toDF("year", "features")
    
    myDF.take(5).foreach(println)
//    myDf.take(5).foreach(println)
//    val rawDF = rdd.toDF("rawDF").cache()
    
//    val obsDF: DataFrame = rdd.toDF("rawDF").cache()
    

    //Step1: tokenize each row
//    val regexTokenizer = new RegexTokenizer()
//    .setInputCol("rawDF")
//    .setOutputCol("prediction")
//    .setPattern(",")
    
    
     
    //set the required paramaters
//    val learningAlg = new LinearRegression().(???)
//    //set appropriate stages
//    val task3Pipeline = new Pipeline().(???)
//    //fit on the training data
//    val task3PipelineModel = task3Pipeline.() //get model summary and print RMSE
//    val task3ModelSummary =
//      task3Pipeline.stage(???).asInstanceOf[LinearRegressionModel].(???) println(???)
//    //make predictions on testing data
//    val predictions = task3PipelineModel.(???)

//print predictions
    val maxIter = 50
    val regularization = 0.1
    val learningRate = 0.1
    
    val myLR = new LinearRegression()
    .setMaxIter(maxIter)
    .setRegParam(regularization)
    .setElasticNetParam(learningRate)
    
//    val lrStage = 
//    val pipeline = ???
//    val pipelineModel: PipelineModel = ???
//    val lrModel = pipelineModel.stages(lrStage).asInstanceOf[LinearRegressionModel]

    //print rmse of our model
    //do prediction - print first k
  }
}