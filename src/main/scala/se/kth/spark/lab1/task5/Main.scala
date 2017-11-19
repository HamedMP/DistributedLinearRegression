package se.kth.spark.lab1.task5

import org.apache.spark._
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.feature.{HashingTF, RegexTokenizer, Tokenizer, VectorSlicer}
import se.kth.spark.lab1.{Array2Vector, DoubleUDF, Vector2DoubleUDF}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.{col, min}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.feature.PolynomialExpansion

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

    val regexTokenizer = new RegexTokenizer()
      .setInputCol("rawDF")
      .setOutputCol("data")
      .setPattern(",")

    val tokensArray = regexTokenizer.transform(train)

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

    val expfeatures = new PolynomialExpansion().setInputCol("features").setOutputCol("expandedfeatures")

    val learningRate = 0.1
    val learningAlg = new LinearRegression()
    .setFeaturesCol("features")
    .setLabelCol("label")
    .setElasticNetParam(learningRate)

    val lrStage = 7
    val pipeline = new Pipeline().setStages(Array(regexTokenizer,arr2Vect,lSlicer,v2d,lShifter,fSlicer,expfeatures,learningAlg))
    val pipelineModel: PipelineModel = pipeline.fit(train)

    val paramGrid = new ParamGridBuilder()
    .addGrid(learningAlg.regParam,Array(0.1,0.9))
    .addGrid(learningAlg.maxIter,Array(10,50))
    .build()
    
    val cv = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(new RegressionEvaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(3)
        
    val cvModel = cv.fit(train)
    val lrModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(lrStage).asInstanceOf[LinearRegressionModel]
    
    //print rmse of our model
    println(s"RMSE: ${lrModel.summary.rootMeanSquaredError}")
    //do prediction - print first k
    cvModel.bestModel.transform(test).select("label", "prediction").take(5).foreach(println)

  }
}