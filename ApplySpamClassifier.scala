package cwm.spam;

import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.rogach.scallop._

class ApplySpamClassifierConf(args: Seq[String]) extends ScallopConf(args) {
  mainOptions = Seq(input, model, output)
  val input = opt[String](descr = "input path", required = true)
  val model = opt[String](descr = "model", required = true)
  val output = opt[String](descr = "output", required = true)
  verify()
}

object ApplySpamClassifier {
  val log = Logger.getLogger(getClass().getName())
  
  // Scores a document based on its list of features.
  def spamminess(features: Array[Int], w: Map[Int, Double]) : Double = {
    var score = 0d
    features.foreach(f => if (w.contains(f)) score += w(f))
    score
  }

  def main(argv: Array[String]) {
    val args = new ApplySpamClassifierConf(argv)

    log.info("Input: " + args.input())
    log.info("Model: " + args.model())
    log.info("Output: " + args.output())

    val conf = new SparkConf().setAppName("Apply Spam Classifier")
    val sc = new SparkContext(conf)

    val outputDir = new Path(args.output())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    val textFile = sc.textFile(args.input(), 1)
    val model = sc.textFile(args.model())

    val lines = textFile.count()

    val w = model.map(line => {
      val columns = line.stripPrefix("(").stripSuffix(")").split(",")
      val key = columns(0).trim.toInt
      val value = columns(1).trim.toDouble
      (key, value)
    }).collectAsMap().toMap

    // Broadcast the model
    val wBroadcast = sc.broadcast(w)

    val classified = textFile.map(line => {
      val columns = line.split(" ")

      val docid = columns(0)
      val label = columns(1)

      val features = columns.slice(2, columns.length).map(_.toInt)
      val spamScore = spamminess(features, wBroadcast.value)

      val prediction = if (spamScore > 0) "spam" else "ham"

      (docid, label, spamScore, prediction)
    })

    classified.saveAsTextFile(args.output())
  }
}

