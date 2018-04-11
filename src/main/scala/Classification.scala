import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

object Classification {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  System.setProperty("hadoop.home.dir","c:/winutil")

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .master("local[*]")
      .getOrCreate()


    val bInput=spark.read.parquet("src/test/data/binary-classification")
      .selectExpr("features","cast(label as Double) as label")
    bInput.show()

    //logistic regression
    val lr=new LogisticRegression()
    //println(lr.explainParams())//see all parameters
    val lrModel = lr.fit(bInput)
    println(lrModel.coefficientMatrix)
    println(lrModel.intercept)

    val summary=lrModel.summary
    val bSummary=summary.asInstanceOf[BinaryLogisticRegressionSummary]
    println(bSummary.areaUnderROC)
    bSummary.roc.show()
    bSummary.pr.show()


    //decision tree
    val dt=new DecisionTreeClassifier()
    val dtmodel=dt.fit(bInput)
    println(s"feature importances for decision tree")
    println(dtmodel.featureImportances)


    //random forest
    val rf=new RandomForestClassifier()
    val rfmodel=rf.fit(bInput)
    println(s"feature importances of random forest")
    println(rfmodel.featureImportances)

    //GBDT
    val gbdt=new GBTClassifier()
    val gbdtmodel=gbdt.fit(bInput)
    println(s"feature importances for GBDT")
    println(gbdtmodel.featureImportances)

    //Naive Bayes
    val nb=new NaiveBayes()
    val nbmodel=nb.fit(bInput.where("label!=0"))

    //evaluation
    val predictAndTest=nbmodel.transform(bInput).select("prediction","label")
      .rdd.map(x=>(x(0).asInstanceOf[Double],x(1).asInstanceOf[Double]))
    val metrics=new BinaryClassificationMetrics(predictAndTest)
    println(metrics.areaUnderROC())

    spark.stop()
  }
}
