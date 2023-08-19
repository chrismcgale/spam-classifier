package cwm.spam;

import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.rogach.scallop._

import scala.collection.mutable.Map
import scala.math.exp
import scala.util.Random

class TrainSpamClassifierConf(args: Seq[String]) extends ScallopConf(args) {
  mainOptions = Seq(input, model)
  val input = opt[String](descr = "input path", required = true)
  val model = opt[String](descr = "model", required = true)
  val shuffle = opt[Boolean](descr = "shuffle")
  verify()
}

object TrainSpamClassifier {
  val log = Logger.getLogger(getClass().getName())

  def main(argv: Array[String]) {
    val args = new TrainSpamClassifierConf(argv)

    log.info("Input: " + args.input())
    log.info("Model: " + args.model())
    log.info("Shuffle: " + args.shuffle())

    val conf = new SparkConf().setAppName("Train Spam Classifier")
    val sc = new SparkContext(conf)

    val outputDir = new Path(args.model())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    val shuffle = args.shuffle()

    val textFile = sc.textFile(args.input(), 1)

    val lines = textFile.count()

    // This is the main learner:
    val delta = 0.002

    val trained = textFile.map(line => {
      val columns = line.split(" ")

      val docid = columns(0)
      val isSpam = if (columns(1) == "spam") 1 else 0
      val features = columns.slice(2, columns.length).map(_.toInt)
      val key = if (shuffle) Random.nextDouble() else 0
  
      (key, (docid, isSpam, features))
    })

    val shuffled = if (args.shuffle()) {
      trained.sortByKey().map{ case (_, v) => (0, v) }.groupByKey(1)
    } else {
      trained.groupByKey(1)
    }

    val updatedWeights = shuffled.map {
      case (_, iter) => {
        // w is the weight vector (make sure the variable is within scope)
        var w = Map[Int, Double]()

        // Scores a document based on its list of features.
        def spamminess(features: Array[Int]) : Double = {
            var score = 0d
            features.foreach(f => if (w.contains(f)) score += w(f))
            score
        }

        iter.foreach {
          case (docid, isSpam, features) => {
          
          // Update the weights as follows:
          val score = spamminess(features)
          val prob = 1.0 / (1 + exp(-score))
          features.foreach(f => {
              if (w.contains(f)) {
                  w(f) += (isSpam - prob) * delta
              } else {
                  w(f) = (isSpam - prob) * delta
              }
           })
         }
        }
        w
      }
    }.reduce((m1, m2) => m1 ++ m2.map { case (k, v) => k -> (v + m1.getOrElse(k, 0.0)) })

    // Convert the updatedWeights map to an RDD
    val mapRDD = sc.parallelize(Seq(updatedWeights), numSlices = 1)

    // Save the RDD as a text file
    mapRDD.flatMap(_.toSeq).map { case (k, v) => s"($k, $v)" }.saveAsTextFile(args.model())
  }
}

