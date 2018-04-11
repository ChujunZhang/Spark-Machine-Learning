import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator,ParamGridBuilder}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans

object Regression {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  System.setProperty("hadoop.home.dir","c:/winutil")

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .master("local[*]")
      .getOrCreate()


    val df=spark.read.load("src/test/data/regression")


    //linear regression
    val lr=new LinearRegression().setMaxIter(10)
      .setRegParam(0.3)//regulazation,like penalty
      .setElasticNetParam(0.8) //control L1 L2
    val lrmodel=lr.fit(df)

    val summary=lrmodel.summary
    summary.residuals.show()
    println(summary.objectiveHistory.toSeq)
    println(s"RMSE:${summary.rootMeanSquaredError}")
    println(s"R-squared: ${summary.r2}")

    //generalized linear regression
    val glr=new GeneralizedLinearRegression()
      .setFamily("gaussian")
      .setLink("identity")
      .setMaxIter(10)
      .setLinkPredictionCol("linkOut")
    val glrmodel=glr.fit(df)

    //evaluation and tuning
    val pipeline=new Pipeline().setStages(Array(glr))
    val params=new ParamGridBuilder()
      .addGrid(glr.regParam,Array(0,0.5,1))
      .build()
    val evaluator=new RegressionEvaluator()
      .setMetricName("rmse")
      .setPredictionCol("prediction")
      .setLabelCol("label")
    val cv=new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(params)
      .setNumFolds(2)
    val model=cv.fit(df)


    val out = model.transform(df)
      .select("prediction","label")
      .rdd.map(x=>(x(0).asInstanceOf[Double],x(1).asInstanceOf[Double]))
    val metrics=new RegressionMetrics(out)

    println(s"MSE = ${metrics.meanSquaredError}")
    println(s"RMSE = ${metrics.rootMeanSquaredError}")
    println(s"R-squared = ${metrics.r2}")
    println(s"MAE = ${metrics.meanAbsoluteError}")
    println(s"Explained variance = ${metrics.explainedVariance}")

    //k-means
    val va=new VectorAssembler()
      .setInputCols(Array("Quantity","UnitPrice"))
      .setOutputCol("features")

    val sales=spark.read
      .option("header","true")
      .option("inferschema","true")
      .csv("src/test/data/retail-data/by-day/*.csv")
      .coalesce(1) //the number of partitions
      .where("Description IS NOT NULL")

    val salesTransform=va.transform(sales)

    val km=new KMeans().setK(5)
    val kmmodel=km.fit(salesTransform)
    val kmsummary=kmmodel.summary
    kmsummary.clusterSizes.foreach(println)
    println(s"computeCost: ${kmmodel.computeCost(salesTransform)}")
    println("cluster centers:")
    kmmodel.clusterCenters.foreach(println)


    spark.stop()
  }
}
