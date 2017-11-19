package se.kth.spark.lab1.task3

import org.apache.spark._
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.ml.feature.{HashingTF, RegexTokenizer, Tokenizer, VectorSlicer}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.functions.{col, min}
import se.kth.spark.lab1.{Array2Vector, DoubleUDF, Vector2DoubleUDF}

object Main {
  def main(args: Array[String]) {

    val sc = SparkSession.builder.appName("lab1").master("local").getOrCreate()
    import sc.implicits._
    
    val filePath = "src/main/resources/millionsong.txt"
    val rdd = sc.sparkContext.textFile(filePath)
    val obsDF: DataFrame = rdd.toDF("rawDF").cache()
    val splits = obsDF.randomSplit(Array(0.8, 0.2))
    val train = splits(0).cache()
    val test = splits(1).cache()
    
    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("rawDF")
      .setOutputCol("data")
      .setPattern(",")

    //Step2: transform with tokenizer and show 5 rows
    val tokensArray = regexTokenizer.transform(train)

    tokensArray.select("data").take(5).foreach(println)

    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector()
      .setInputCol("data")
      .setOutputCol("arr2VectPrediction")

    val arr2VectOut = arr2Vect.transform(tokensArray)

    //  Step4: extract the label(year) into a new column
    val lSlicer = new VectorSlicer().setInputCol("arr2VectPrediction").setOutputCol("yearArray")
    lSlicer.setIndices(Array(0))
    val output = lSlicer.transform(arr2VectOut)


    //    Step5: convert type of the label from vector to double (use our Vector2Double)
    val v2d = new Vector2DoubleUDF(v => v(0).toDouble)
      .setInputCol("yearArray")
      .setOutputCol("year")

    val v2dOut = v2d.transform(output) // .setInputCol("year").setOutputCol("yearDouble")

    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF)
    val minYear = v2dOut.agg(min(col("year"))).first().getDouble(0)
    val lShifter = new DoubleUDF(year => year - minYear)
      .setInputCol("year")
      .setOutputCol("label")
    //      .transform(v2d)

    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer().setIndices(Array(1, 2, 3))
      .setInputCol("arr2VectPrediction")
      .setOutputCol("features")

    val maxIter = 50
    val regularization = 0.1
    val learningRate = 0.1


    //set the required paramaters
    val learningAlg = new LinearRegression()
      .setMaxIter(maxIter)
      .setRegParam(regularization)
      .setElasticNetParam(learningRate)
      .setLabelCol("label")
      .setFeaturesCol("features")

    //set appropriate stages
    //Step8: put everything together in a pipeline
    val pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, learningAlg))

//    //fit on the training data
    val pipelineModel = pipeline.fit(train)
//
//    //Step10: transform data with the model - do predictions TODO
    val transformed = pipelineModel.transform(test)

    //Step11: drop all columns from the dataframe other than label and features
//    val finalDF = transformed.select("label", "features")

    val trainingSummary = pipelineModel.stages(6).asInstanceOf[LinearRegressionModel].summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")


//    (???) println (???)
    //make predictions on testing data
//    val predictions = pipelineModel.(???)

//print predictions

    
//    val lrStage = 
//    val pipeline = ???
//    val pipelineModel: PipelineModel = ???
//    val lrModel = pipelineModel.stages(lrStage).asInstanceOf[LinearRegressionModel]

    //print rmse of our model
    //do prediction - print first k
  }
}