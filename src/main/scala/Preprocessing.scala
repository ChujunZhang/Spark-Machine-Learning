import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.feature.SQLTransformer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.ml.feature.QuantileDiscretizer
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.feature.{ChiSqSelector, Tokenizer}


object Preprocessing {

  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)

  def loadSalesData(spark:SparkSession,path:String)={
    spark.read
      .option("header","true")
      .option("inferschema","true")
      .csv(path)
      .coalesce(5) //the number of partitions
      .where("Description IS NOT NULL")
  }

  //Tokenizer
  def seperateColumn(df:DataFrame,col:String)={ //seperate the string in a column, one kind of transformer
    val tkn = new Tokenizer().setInputCol(col)
    tkn.transform(df.select(col))
  }


  //Tokenizer using regular expression
  def regexSeperateColumn(df:DataFrame)={
    val rt=new RegexTokenizer()
      .setInputCol("Description")
      .setOutputCol("DescOut")
      .setPattern(" ") //what is the delimiter
      .setToLowercase(true)

    rt.transform(df.select("Description")).show()
  }


  //StandardScaler
  def scaling(df:DataFrame,col:String)={//scale, one kind of estimators
    val ss = new StandardScaler().setInputCol(col).setOutputCol("scalingCol")
    ss.fit(df).transform(df)
  }

  //RFormula
  def manipulate(df:DataFrame,formula:String)={
    //change Y to 1 or 0, create new interaction variables
    val superivsed=new RFormula().setFormula(formula)
    superivsed.fit(df).transform(df)
  }

  //SQLTransformer
  def sqltransform(df:DataFrame)={
    val sql=new SQLTransformer().setStatement(
      """
        select sum(Quantity),count(*),CustomerID
        from __THIS__
        group by CustomerID
      """)
    sql.transform(df)
  }

  //VectorAssembler
  def bigVector(df:DataFrame)={
    val va=new VectorAssembler().setInputCols(Array("int1","int2","int3"))
    va.transform(df)
  }

  //bucketing  (only handle double)
  def bucketing(spark:SparkSession)={
    val contDF=spark.range(20).selectExpr("cast(id as double)")

    //decide the bucket range
    val bucketBorders=Array(-1.0,5.0,10.0,250.0,600.0)
    val bucketer=new Bucketizer()
      .setSplits(bucketBorders)
      .setInputCol("id")
      .setOutputCol("bucket")
    println(s"bucket with decided borders:")
    bucketer.transform(contDF).show()

    //use quantile for bucket
    val bucketer2=new QuantileDiscretizer()
      .setNumBuckets(5)
      .setInputCol("id")
      .setOutputCol("bucket")
    val fittedbucketer=bucketer2.fit(contDF)
    fittedbucketer.transform(contDF).show()
  }

  //normalization
  def normalization(df:DataFrame)={
    val normalzation=new Normalizer()
      .setP(1)
      .setInputCol("features")
      .setOutputCol("normalized features")
    normalzation.transform(df).show()
  }

  //categorical to index
  def indexer(df:DataFrame)={
    val indexer=new StringIndexer()
      .setInputCol("lab")
      .setOutputCol("labIndexer")
      .setHandleInvalid("skip") //a new index not in training, just skip the entire row
    println("to Index:")
    val ind=indexer.fit(df).transform(df)
    ind.show()

    val reverserToLabel = new IndexToString()
      .setInputCol("labIndexer")
      .setOutputCol("tolabel")
    println("back to string:")
    reverserToLabel.transform(ind).show()
  }

  //indexing vectors
  //use this when you have a vector features with both categorical and numeric
  //you want to change categorical to zero-based index
  //use maxCategories to tell, less than this distinct values are categorical feature, need to convert
  def indexingVectors(spark:SparkSession)={
    val idxIn = spark.createDataFrame(
      Seq(
        (Vectors.dense(1,2,3),1),
        (Vectors.dense(2,5,6),2),
        (Vectors.dense(1,8,9),3)
      )).toDF("features","label")
    val indexr=new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("featureIndex")
      .setMaxCategories(2)
    indexr.fit(idxIn).transform(idxIn).show()
  }


  // remove some common words
  def removeWords(df:DataFrame)={
    val rt=new Tokenizer()
      .setInputCol("Description")
      .setOutputCol("DescOut")

    val tokenized=rt.transform(df.select("Description"))

    val englishStopWords = StopWordsRemover.loadDefaultStopWords("english")
    val stops=new StopWordsRemover()
      .setInputCol("DescOut")
      .setStopWords(englishStopWords)

    stops.transform(tokenized).show()
  }

  //PCA
  def pca(df:DataFrame)={
    val pca=new PCA().setInputCol("features").setK(2)
    pca.fit(df).transform(df).show()
  }

  // polunomial expansion like 2 degree interaction
  def polynomial(df:DataFrame)={
    val pe=new PolynomialExpansion().setInputCol("features").setDegree(2)
    pe.transform(df).show()
  }

  //feature selection
  def chisquareSelector(df:DataFrame)={
    val tkn=new Tokenizer()
      .setInputCol("Description")
      .setOutputCol("DescOut")
    val tokenized=tkn
      .transform(df.select("Description","CustomerId"))
      .where("CustomerId IS NOT NULL")
    val chisq=new ChiSqSelector()
      .setFeaturesCol("countVec")
      .setLabelCol("CustomerId")
      .setNumTopFeatures(2)
    chisq.fit(tokenized).transform(tokenized).show()
  }
}

