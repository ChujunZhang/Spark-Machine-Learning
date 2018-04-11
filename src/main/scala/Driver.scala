import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}


object Driver {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  System.setProperty("hadoop.home.dir","c:/winutil")

  def main(args: Array[String]): Unit = {

    // Setup SparkSession
    val spark = SparkSession.builder
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .master("local[*]")
      .getOrCreate()

    val fakeIntDF=spark.read.parquet("src/test/data/simple-ml-integers")
    val simpleDF = spark.read.json("src/test/data/simple-ml")
    val scaleDF = spark.read.parquet("src/test/data/simple-ml-scaling")
    val sales=Preprocessing.loadSalesData(spark,"src/test/data/retail-data/by-day/*.csv")

    sales.cache()
    sales.show()
    scaleDF.show()
    simpleDF.show()

    Preprocessing.seperateColumn(sales,"Description").show()
    Preprocessing.regexSeperateColumn(sales)
    Preprocessing.scaling(scaleDF,"features").show()

    val formula="lab ~. + color:value1+color:value2" //. is all the features,: is interaction
    Preprocessing.manipulate(simpleDF,formula).show()

    Preprocessing.sqltransform(sales).show()
    Preprocessing.bigVector(fakeIntDF).show()

    Preprocessing.bucketing(spark)
    Preprocessing.normalization(scaleDF)
    Preprocessing.indexer(simpleDF)
    Preprocessing.indexingVectors(spark)
    Preprocessing.removeWords(sales)//like "of" is removed

    Preprocessing.pca(scaleDF)
    Preprocessing.polynomial(scaleDF)
    Preprocessing.chisquareSelector(sales)



    spark.stop()
  }


}
