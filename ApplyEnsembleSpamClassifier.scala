package cwm.spam;

import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.rogach.scallop._

class ApplyEnsembleSpamClassifierConf(args: Seq[String]) extends ScallopConf(args) {
  mainOptions = Seq(input, model, output)
  val input = opt[String](descr = "input path", required = true)
  val model = opt[String](descr = "model", required = true)
  val output = opt[String](descr = "output", required = true)
  val method = opt[String](descr = "method", required = true)
  verify()
}

object ApplyEnsembleSpamClassifier {
  val log = Logger.getLogger(getClass().getName())

  def processModel(model: org.apache.spark.rdd.RDD[String]): Map[Int, Double] = {
    model.map(line => {
      val columns = line.stripPrefix("(").stripSuffix(")").split(",")
      val key = columns(0).trim.toInt
      val value = columns(1).trim.toDouble
      (key, value)
    }).collectAsMap().toMap
  }

  def vote(features: Array[Int], w: Map[Int, Double]) : Double = {
    var score = 0d
    features.foreach(f => if (w.contains(f)) score += w(f))
    score
  }

  // Scores a document based on its list of features.
  def spamminess(features: Array[Int], method: String, w1: Map[Int, Double], w2: Map[Int, Double], w3: Map[Int, Double]) : Double = {
    var score = 0d
    if (method == "average") {
      features.foreach(f => score += (w1.getOrElse(f, 0.0) + w2.getOrElse(f, 0.0) + w3.getOrElse(f, 0.0)) / 3)
    } else if (method == "vote") {
      val v1 = if (vote(features, w1) > 0) 1 else -1
      val v2 = if (vote(features, w2) > 0) 1 else -1
      val v3 = if (vote(features, w3) > 0) 1 else -1
      score = v1 + v2 + v3
    } else {
      throw new Error("Invalid ensemble method")
    }
    score
  }

  def main(argv: Array[String]) {
    val args = new ApplyEnsembleSpamClassifierConf(argv)

    log.info("Input: " + args.input())
    log.info("Model: " + args.model())
    log.info("Output: " + args.output())
    log.info("Method: " + args.method())

    val conf = new SparkConf().setAppName("Apply Spam Classifier")
    val sc = new SparkContext(conf)

    val outputDir = new Path(args.output())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    val textFile = sc.textFile(args.input(), 1)
    val model1 = sc.textFile(args.model() + "/part-00000" )
    val model2 = sc.textFile(args.model() + "/part-00001" )
    val model3 = sc.textFile(args.model() + "/part-00002" )

    val method = args.method()

    val lines = textFile.count()

    val w1 = processModel(model1)
    val w2 = processModel(model2)
    val w3 = processModel(model3)

    val w1Broadcast = sc.broadcast(w1)
    val w2Broadcast = sc.broadcast(w2)
    val w3Broadcast = sc.broadcast(w3)

    val classified = textFile.map(line => {
      val columns = line.split(" ")

      val docid = columns(0)
      val label = columns(1)

      val features = columns.slice(2, columns.length).map(_.toInt)
      val spamScore = spamminess(features, method, w1Broadcast.value, w2Broadcast.value, w3Broadcast.value)

      val prediction = if (spamScore > 0) "spam" else "ham"

      (docid, label, spamScore, prediction)
    })

    classified.saveAsTextFile(args.output())
  }
}

