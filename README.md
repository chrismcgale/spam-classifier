# Spam Classifier #

The basic idea for this project comes from the work described in "Efficient and Effective Spam Filtering and Re-ranking for Large Web Datasets" by Cormack, Smucker, and Clarke. It applies a stochastic gradient descent to progressively increase performance. The paper defines "spam" pages as "those which are unlikely to be judged relevant to any query that might reasonably retrieve them".


## Requirements ##
Java \
Scala \
Scala build Tool (sbt) \
Maven \
gcc

## Data ## 
We'll be using a partial copy of the ClueWeb09 dataset. A 5TB, compressed (25TB uncompressed) behemoth, comprising over 1 billion websites in 10 languages. 

## Steps ##
1. Compile

```mvn clean package```

2. Train 

Add data shuffling with the `--shuffle` option.

Replace 'spam.train.group_x.txt' with your input files.

```spark-submit --driver-memory 2g --class cwm.TrainSpamClassifier \ target/assignments-1.0.jar --input spam.train.group_x.txt --model cwm-model-group_x```

3. Apply

```spark-submit --driver-memory 2g --class cwm.ApplySpamClassifier target/assignments-1.0.jar --input spam.test.qrels.txt --output cwm-test-group_x --model cwm-model-group_x```

4. Evaluate

    1. Compile compute_spam.c `gcc -O2 -o compute_spam_metrics compute_spam_metrics.c -lm`

    2. Run ```$ ./spam_eval.sh cwm-test-group_x```

5. Train an ensemble model

    1. Move models into the same directory

    ```mkdir cwm-model-fusion \ cp cwm-model-group_x/part-00000 cwm-model-fusion/part-00000 \ cp cwm-model-group_y/part-00000 cwm-model-fusion/part-00001 \ cp cwm-model-britney/part-00000 cwm-model-fusion/part-00002```

    2. Run ensemble model. The two methods available  are: score 'average' and 'voting'.

    ```spark-submit --driver-memory 2g --class cwm.ApplyEnsembleSpamClassifier \ target/assignments-1.0.jar --input spam.test.qrels.txt \ --output -est-fusion-average --model model-fusion --method average```

## My Results ##

By applying a shuffled ensembled classifier I achieved `1-ROCA%: 13.81` on the 766 test-britney found in the above paper. Training time 1 minute 40s, applying time 74s.

## Next steps ##
1. Implement other models / boosting techniques
2. Set up automatic deployment of spaek cluster with ansible
