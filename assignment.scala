package assignment19

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{Column, DataFrame, SparkSession}

import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.collection.immutable.Range
import org.apache.spark.sql.functions._

/* 
 * Assignment 1 Scala, 
 * Group Members: manojprabhakar_parthasarathy@tuni.fi, chanukya_pekala@tuni.fi
 * The code contribution has been done equally and merge has been done by chanukya_pekala
*/

object assignment {

  Logger.getLogger("org").setLevel(Level.OFF)

  val spark = SparkSession.builder()
    .appName("ex2")
    .config("spark.driver.host", "localhost")
    .master("local")
    .getOrCreate()

  val dataK5D2 = spark
    .read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", true)
    .option("delimiter", ",")
    .csv("data/dataK5D2.csv")

  dataK5D2.printSchema

  val dataK5D3 = spark
    .read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", true)
    .option("delimiter", ",")
    .csv("data/dataK5D3.csv")

  dataK5D3.printSchema
  val dataK5D3WithLabels = dataK5D2.withColumn("labels", when(col("LABEL") === " Ok", 1).otherwise(2));

  /*
   * K-Means clustering on two Dimensional Data
   * The Function takes a DataFrame and k
   * The function returns the k 2-dimensional cluster centers of the classes
  */
  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {
    val kMeans = new KMeans

    // Cluster the data into two classes using KMeans
    val numClusters = k
    val numIterations = 20
    val vectorAssembler = new VectorAssembler() .setInputCols(Array("a", "b")) .setOutputCol("features")
    val transformationPipeline = new Pipeline().setStages(Array(vectorAssembler))
    val pipeLine = transformationPipeline.fit(df)
    val transformedTraining = pipeLine.transform(df)
    transformedTraining.show()

    val kmeans = new KMeans().setK(numClusters).setSeed(1L)
    val kmModel = kmeans.fit(transformedTraining)
    kmModel.summary.predictions.show()
    val pred = kmModel.summary.predictions.collect()
    pred.foreach(println)

    kmModel.clusterCenters.foreach(println)
    val clusterArrays = new Array[(Double,Double)](k)

    var i = 0
    kmModel.clusterCenters.foreach(f => {
      var f1 = f.toArray
      clusterArrays(i) = (f1(0), f1(1))
      i = i+1
    })

    clusterArrays
  }
  
  
  /*
   * K-Means clustering on Three Dimensional Data
   * The Function takes a DataFrame and k
   * The function returns the k 3-dimensional cluster centers of the classes 
  */
  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {
    val kMeans = new KMeans

    // Cluster the data into two classes using KMeans
    val numClusters = k
    val numIterations = 20
    val vectorAssembler = new VectorAssembler() .setInputCols(Array("a", "b", "c")) .setOutputCol("features")
    val transformationPipeline = new Pipeline().setStages(Array(vectorAssembler))
    val pipeLine = transformationPipeline.fit(df)
    val transformedTraining = pipeLine.transform(df)
    transformedTraining.show()

    val kmeans = new KMeans().setK(numClusters).setSeed(1L)
    val kmModel = kmeans.fit(transformedTraining)


    kmModel.summary.predictions.show()


    kmModel.clusterCenters.foreach(println)
    val clusterArrays = new Array[(Double,Double,Double)](k)

    var i = 0
    kmModel.clusterCenters.foreach(f => {
      var f1 = f.toArray
      clusterArrays(i) = (f1(0), f1(1), f1(2))
      i = i+1
    })
    kmModel.summary.predictions.show

    clusterArrays
  }

  /*
   * K-Means clustering on Three Dimensional Data, 3rd Column is label
   * The Function takes a DataFrame and k
   * The function returns The clusters centers of classes whose conditions are Fatal
   * E.g. the 3rd dimension column is of label 1 and 2 where 1 is Ok, 2 is Fatal
   * If a clusterCenter exist in the third dimension such that if it's 3rd dimentions is 
   * greater than 1.5 then their condition is decided to be Fatal and the clusterCentroid is returned
  */
  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {
    val vectorAssembler = new VectorAssembler().setInputCols(Array("a", "b", "labels")).setOutputCol("features")
    val transformationPipeline = new Pipeline().setStages(Array(vectorAssembler))
    val pipeLine = transformationPipeline.fit(df)
    val transformedTraining = pipeLine.transform(df)
    transformedTraining.show()
    val kmeans = new KMeans().setK(k).setSeed(1L)
    val kmModel = kmeans.fit(transformedTraining)
    kmModel.summary.predictions.show(1, false)
    val clusterArrays = new Array[(Double, Double)](k)
    var i = 0
    kmModel.clusterCenters.foreach(f => {
      var f1 = f.toArray
      if (f1(2) > 1.5) {
        clusterArrays(i) = (f1(0), f1(1))
        i = i + 1
      }
    }
    )
    clusterArrays.slice(0, i)
  }

  /*
   * The Elbow Method, Finding the optimum number of k clusters
   */
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)] = {
    var i = 0
    val kCost = new Array[(Int, Double)](high - low + 1)
    for (k <- low to high) {
      val numClusters = k
      val numIterations = 20
      val vectorAssembler = new VectorAssembler().setInputCols(Array("a", "b")).setOutputCol("features")
      val transformationPipeline = new Pipeline().setStages(Array(vectorAssembler))
      val pipeLine = transformationPipeline.fit(df)
      val transformedTraining = pipeLine.transform(df)
      val kmeans = new KMeans().setK(numClusters).setSeed(1L)
      val kmModel = kmeans.fit(transformedTraining)
      val WSSE = kmModel.computeCost(transformedTraining)
      kCost(i) = (k, WSSE)
      i += 1
    }
    kCost
  }
}

