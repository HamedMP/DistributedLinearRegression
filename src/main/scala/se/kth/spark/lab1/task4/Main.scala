package se.kth.spark.lab1.task4

import org.apache.spark._
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.{HashingTF, RegexTokenizer, Tokenizer, VectorSlicer}
import se.kth.spark.lab1.{Array2Vector, DoubleUDF, Vector2DoubleUDF}
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.{col, min}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._
    
    val filePath = "src/main/resources/millionsong.txt"
    val rdd = sc.textFile(filePath)

    val obsDF: DataFrame = rdd.toDF("rawDF").cache()
    
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("rawDF")
      .setOutputCol("data")
      .setPattern(",")
      
    val tokensArray = regexTokenizer.transform(obsDF)
    
    val arr2Vect = new Array2Vector()
      .setInputCol("data")
      .setOutputCol("arr2VectPrediction")
     
    val arr2VectOut = arr2Vect.transform(tokensArray)
    
    val lSlicer = new VectorSlicer().setInputCol("arr2VectPrediction").setOutputCol("yearArray")
    lSlicer.setIndices(Array(0))
    val output = lSlicer.transform(arr2VectOut)

    val v2d = new Vector2DoubleUDF(v => v(0).toDouble)
      .setInputCol("yearArray")
      .setOutputCol("year")

    val v2dOut = v2d.transform(output) // .setInputCol("year").setOutputCol("yearDouble")

    val minYear = v2dOut.agg(min(col("year"))).first().getDouble(0)
    val lShifter = new DoubleUDF(year => year - minYear)
      .setInputCol("year")
      .setOutputCol("label")

    val fSlicer = new VectorSlicer().setIndices(Array(1, 2, 3))
      .setInputCol("arr2VectPrediction")
      .setOutputCol("features")

      
    val learningRate = 0.1

    //set the required paramaters
    val learningAlg = new LinearRegression()
    .setFeaturesCol("features")  
    .setLabelCol("label")
    .setElasticNetParam(learningRate)
      
    val pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, learningAlg))

    val pipelineModel = pipeline.fit(obsDF)

    val transformed = pipelineModel.transform(obsDF)

    val lrStage = 6 //index of our linearRegression algorithm
    val trainingSummary = pipelineModel.stages(lrStage).asInstanceOf[LinearRegressionModel].summary

    val paramGrid = new ParamGridBuilder()
    .addGrid(learningAlg.regParam,Array(0.1,0.9))
    .addGrid(learningAlg.maxIter,Array(10,50))
    .build()
    
    val cv = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(new RegressionEvaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(3)
        
    val cvModel = cv.fit(obsDF)
    val lrModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(lrStage).asInstanceOf[LinearRegressionModel]
    
    //print rmse of our model
    println(s"RMSE: ${lrModel.summary.rootMeanSquaredError}")
    //do prediction - print first k
    cvModel.bestModel.transform(obsDF).select("label", "prediction").take(5).foreach(println)
  }
}