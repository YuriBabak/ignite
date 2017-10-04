/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.ignite.ml.encog;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.apache.ignite.IgniteDataStreamer;
import org.apache.ignite.internal.util.IgniteUtils;
import org.apache.ignite.ml.Model;
import org.apache.ignite.ml.encog.caches.TestTrainingSetCache;
import org.apache.ignite.ml.encog.evolution.operators.IgniteEvolutionaryOperator;
import org.apache.ignite.ml.encog.evolution.operators.NodeCrossover;
import org.apache.ignite.ml.encog.evolution.operators.WeightCrossover;
import org.apache.ignite.ml.encog.metaoptimizers.AddLeaders;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.apache.ignite.ml.math.functions.IgniteSupplier;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.genetic.MLMethodGenome;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.TrainingSetScore;
import org.jetbrains.annotations.NotNull;

/** IMPL NOTE do NOT run this as test class because JUnit3 will pick up redundant test from superclass. */
public class PredictionTracer extends GenTest {
    /** */
    private static final String MNIST_LOCATION = "C:/work/test/mnist/";

    /** */
    public void testPrediction() throws IOException {
        IgniteUtils.setCurrentIgniteName(ignite.configuration().getIgniteInstanceName());

        MnistUtils.Pair<double[][], double[][]> mnist = loadMnist();

        // create training data
        IgniteSupplier<IgniteNetwork> fact = this::supplyFact;

        List<IgniteEvolutionaryOperator> evoOps = Arrays.asList(
            new NodeCrossover(0.5, "nc"),
            new WeightCrossover(0.5, "wc")
            );

        GaTrainerCacheInput<IgniteNetwork, MLMethodGenome, MLMethodGenome> input = new GaTrainerCacheInput<>(TestTrainingSetCache.NAME,
            fact,
            mnist.getFst().length,
            60,
            evoOps,
            30,
            (in, ignite) -> new TrainingSetScore(in.mlDataSet(ignite)),
            3,
            new AddLeaders(0.2),
            0.02
            );

        @SuppressWarnings("unchecked")
        EncogMethodWrapper mdl = new GATrainer(ignite).train(input);

        calculateError(mdl);
    }

    /** */
    @NotNull private IgniteNetwork supplyFact() {
        IgniteNetwork res = new IgniteNetwork();
        addLayers(res);
        res.getStructure().finalizeStructure();

        res.reset();

        return res;
    }

    /** */
    private void addLayers(IgniteNetwork res) {
        res.addLayer(new BasicLayer(null,true,28 * 28));
        res.addLayer(new BasicLayer(new org.encog.engine.network.activation.ActivationSigmoid(),true,50));
        res.addLayer(new BasicLayer(new org.encog.engine.network.activation.ActivationSoftMax(),false,10));
    }

    /** */
    private MnistUtils.Pair<double[][], double[][]> loadMnist() throws IOException {
        System.out.println("Reading mnist...");
        MnistUtils.Pair<double[][], double[][]> mnist = MnistUtils.mnist(MNIST_LOCATION + "train-images-idx3-ubyte", MNIST_LOCATION + "train-labels-idx1-ubyte", new Random(), 60_000);
//        MnistUtils.Pair<double[][], double[][]> mnist = MnistUtils.mnist(MNIST_LOCATION + "t10k-images-idx3-ubyte", MNIST_LOCATION + "t10k-labels-idx1-ubyte", new Random(), 10_000);

        System.out.println("Reading mnist done.");

        System.out.println("Loading MNIST into test cache...");

        loadIntoCache(mnist);
        System.out.println("Loading MNIST into test cache done.");
        return mnist;
    }

    /** */
    private void loadIntoCache(MnistUtils.Pair<double[][], double[][]> mnist) {
        TestTrainingSetCache.getOrCreate(ignite);

        try (IgniteDataStreamer<Integer, MLDataPair> stmr = ignite.dataStreamer(TestTrainingSetCache.NAME)) {
            // Stream entries.

            int samplesCnt = mnist.getFst().length;
            System.out.println("Loading " + samplesCnt + " samples into cache...");
            for (int i = 0; i < samplesCnt; i++)
                stmr.addData(i, new BasicMLDataPair(new BasicMLData(mnist.fst[i]), new BasicMLData(mnist.snd[i])));
        }
    }

    /** */
    private void calculateError(EncogMethodWrapper mdl) throws IOException {
        MnistUtils.Pair<double[][], double[][]> testMnistData = MnistUtils.mnist(MNIST_LOCATION + "t10k-images-idx3-ubyte", MNIST_LOCATION + "t10k-labels-idx1-ubyte", new Random(), 10_000);

        IgniteBiFunction<Model<MLData, double[]>, MnistUtils.Pair<double[][], double[][]>, Double> errorsPercentage = errorsPercentage();
        Double accuracy = errorsPercentage.apply(mdl, testMnistData);
        System.out.println(">>> Errs percentage: " + accuracy);
    }

    /** */
    private IgniteBiFunction<Model<MLData, double[]>, MnistUtils.Pair<double[][],double[][]>, Double> errorsPercentage(){
        return (model, pair) -> {

            double[][] k = pair.getFst();
            double[][] v = pair.getSnd();

            assert k.length == v.length;

            long total = 0L;
            long cnt = 0L;
            for (int i = 0; i < k.length; i++) {
                total++;

                double[] predict = model.predict(new BasicMLData(k[i]));
                if (i % 100 == 0)
                    System.out.println(Arrays.toString(predict));

                if(verifyPrediction(v[i], predict))
                    cnt++;
            }

            return 1 - (double)cnt / total;
        };
    }

    /** */
    private boolean verifyPrediction(double[] exp, double[] predict) {
        // todo add output to CSV file here, possibly along with index
        int predictedDigit = toDigit(predict);
        int idealDigit = toDigit(exp);

        return predictedDigit == idealDigit;
    }
}
