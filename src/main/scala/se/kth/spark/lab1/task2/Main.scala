package se.kth.spark.lab1.task2

import org.apache.hadoop.io.nativeio.NativeIO.POSIX.Stat
import se.kth.spark.lab1._
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.{col, min}

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rdd = sc.textFile(filePath)
    val rawDF = rdd.toDF("rawDF").cache()    
    
    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("rawDF")
      .setOutputCol("prediction")
      .setPattern(",")

    //Step2: transform with tokenizer and show 5 rows
    val tokensArray = regexTokenizer.transform(rawDF)

    tokensArray.select("prediction").take(5).foreach(println)
    
    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector()
      .setInputCol("prediction")
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
    
//  using all features for the cluster  
//    val numFeatures = tokensArray.select("predictions").take(1).size
//    val numFeatures = tokensArray.select("predictions").take(1).head.size
//    val fIndicesArrays = (1 to numFeatures).toArray
//    val fsclier = new VectorSlicer().setIndices(fIndicesArrays).setInputCol("arr2VectPrediction")
//      .setOutputCol("features")
// 
 
    //Step8: put everything together in a pipeline
    val pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer))

    //Step9: generate model by fitting the rawDf into the pipeline
    val pipelineModel = pipeline.fit(rawDF)

    //Step10: transform data with the model - do predictions TODO
    val transformed = pipelineModel.transform(rawDF)

    //Step11: drop all columns from the dataframe other than label and features
    val finalDF = transformed.drop("rawDF","prediction","arr2VectPrediction","yearArray","year")
//    val finalDF = transformed.select("label", "features")

    finalDF.show()

  }
}