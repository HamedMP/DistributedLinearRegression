package se.kth.spark.lab1.task6

import org.apache.spark._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{PolynomialExpansion, RegexTokenizer, VectorAssembler, VectorSlicer}
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SparkSession}
import org.apache.spark.ml.PipelineModel
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

//    val expfeatures = new PolynomialExpansion().setInputCol("features").setOutputCol("expandedfeatures")

    val learningRate = 0.1
    val myLR = new MyLinearRegressionImpl()
      .setLabelCol("label")
      .setFeaturesCol("features")

    val learningAlg = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setElasticNetParam(learningRate)

    val lrStage = 6
    val pipeline = new Pipeline().setStages(
      Array(regexTokenizer,
            arr2Vect,
            lSlicer,
            v2d,
            lShifter,
            fSlicer,
            myLR))

    val pipelineModel: PipelineModel = pipeline.fit(train)
    val myLRModel = pipelineModel.stages(lrStage).asInstanceOf[MyLinearModelImpl]

    println(myLRModel.trainingError.deep.mkString("\n"))

    //print rmse of our model
    //do prediction - print first k

    pipelineModel.transform(test).select("label", "prediction").take(5).foreach(println)

  }
}