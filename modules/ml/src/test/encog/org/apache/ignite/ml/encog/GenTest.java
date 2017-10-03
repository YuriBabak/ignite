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
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteDataStreamer;
import org.apache.ignite.configuration.IgniteConfiguration;
import org.apache.ignite.internal.util.IgniteUtils;
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.Model;
import org.apache.ignite.ml.encog.caches.TestTrainingSetCache;
import org.apache.ignite.ml.encog.evolution.operators.IgniteEvolutionaryOperator;
import org.apache.ignite.ml.encog.evolution.operators.MutateNodes;
import org.apache.ignite.ml.encog.evolution.operators.NodeCrossover;
import org.apache.ignite.ml.encog.evolution.operators.WeightCrossover;
import org.apache.ignite.ml.encog.evolution.operators.WeightMutation;
import org.apache.ignite.ml.encog.metaoptimizers.AddLeaders;
import org.apache.ignite.ml.encog.metaoptimizers.LearningRateAdjuster;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.apache.ignite.ml.math.functions.IgniteSupplier;
import org.apache.ignite.testframework.junits.IgniteTestResources;
import org.apache.ignite.testframework.junits.common.GridCommonAbstractTest;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.genetic.MLMethodGenome;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.TrainingSetScore;
import org.junit.Test;

public class GenTest  extends GridCommonAbstractTest {
    public static final String MNIST_LOCATION = "/home/enny/Downloads/";
    private static final int NODE_COUNT = 3;

    /** Grid instance. */
    protected Ignite ignite;

    /**
     * Default constructor.
     */
    public GenTest() {
        super(false);
    }

    /**
     * {@inheritDoc}
     */
    @Override protected void beforeTest() throws Exception {
        ignite = grid(NODE_COUNT);
    }

    /** {@inheritDoc} */
    @Override protected void beforeTestsStarted() throws Exception {
        for (int i = 1; i <= NODE_COUNT; i++)
            startGrid(i);
    }

    /** {@inheritDoc} */
    @Override protected void afterTestsStopped() throws Exception {
        stopAllGrids();
    }

    @Override protected long getTestTimeout() {
        return 6000000;
    }

    @Override protected IgniteConfiguration getConfiguration(String igniteInstanceName,
        IgniteTestResources rsrcs) throws Exception {
        IgniteConfiguration configuration = super.getConfiguration(igniteInstanceName, rsrcs);
        configuration.setIncludeEventTypes();
        configuration.setPeerClassLoadingEnabled(true);
        configuration.setMetricsUpdateFrequency(2000);

        return configuration;
    }

    @Test
    public void test() throws IOException {
        IgniteUtils.setCurrentIgniteName(ignite.configuration().getIgniteInstanceName());

        System.out.println("Reading mnist...");
        MnistUtils.Pair<double[][], double[][]> mnist = MnistUtils.mnist(MNIST_LOCATION + "train-images-idx3-ubyte", MNIST_LOCATION + "train-labels-idx1-ubyte", new Random(), 60_000);
//        MnistUtils.Pair<double[][], double[][]> mnist = MnistUtils.mnist(MNIST_LOCATION + "t10k-images-idx3-ubyte", MNIST_LOCATION + "t10k-labels-idx1-ubyte", new Random(), 10_000);

        System.out.println("Done.");

        System.out.println("Loading MNIST into test cache...");
        loadIntoCache(mnist);
        System.out.println("Done.");

        // create training data
        IgniteSupplier<BasicNetwork> fact = () -> {
            BasicNetwork res = new BasicNetwork();
            res.addLayer(new BasicLayer(null,true,28 * 28));
            res.addLayer(new BasicLayer(new org.encog.engine.network.activation.ActivationSigmoid(),true,50));
            res.addLayer(new BasicLayer(new org.encog.engine.network.activation.ActivationSoftMax(),false,10));
            res.getStructure().finalizeStructure();

            res.reset();
            return res;
        };

        List<IgniteEvolutionaryOperator> evoOps = Arrays.asList(
            new WeightMutation(0.4, 0.05,"wm"),
//            new WeightCrossover(0.5, "wc"),
            new NodeCrossover(0.5, "nc"),
            new MutateNodes(10, 0.2, 0.05, "mn")
//            new Hillclimb(0.4));
        );

        GaTrainerCacheInput<BasicNetwork, MLMethodGenome, MLMethodGenome> input = new GaTrainerCacheInput<>(TestTrainingSetCache.NAME,
            fact,
            mnist.getFst().length,
            60,
            evoOps,
            30,
            (in, ignite) -> new TrainingSetScore(in.mlDataSet(ignite)),
            3,
            new AddLeaders(0.2),//.andThen(new LearningRateAdjuster()),
            0.02
        );

        EncogMethodWrapper model = new GATrainer(ignite).train(input);

        calculateError(model);
    }

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

    private void calculateError(EncogMethodWrapper model) throws IOException {
        MnistUtils.Pair<double[][], double[][]> testMnistData = MnistUtils.mnist(MNIST_LOCATION + "t10k-images-idx3-ubyte", MNIST_LOCATION + "t10k-labels-idx1-ubyte", new Random(), 10_000);

        IgniteBiFunction<Model<MLData, double[]>, MnistUtils.Pair<double[][], double[][]>, Double> errorsPercentage = errorsPercentage();
        Double accuracy = errorsPercentage.apply(model, testMnistData);
        System.out.println(">>> Errs percentage: " + accuracy);
    }

    private IgniteBiFunction<Model<MLData, double[]>, MnistUtils.Pair<double[][],double[][]>, Double> errorsPercentage(){
        return (model, pair) -> {
            long total = 0L;
            long cnt = 0L;

            double[][] k = pair.getFst();
            double[][] v = pair.getSnd();

            assert k.length == v.length;

            for (int i = 0; i < k.length; i++) {
                total++;

                double[] predict = model.predict(new BasicMLData(k[i]));
//                if(!Arrays.equals(predict, v[i]))
//                System.out.println();
                if (i % 100 == 0)
                    System.out.println(Arrays.toString(predict));
                int predictedDigit = toDigit(predict);
                int idealDigit = toDigit(v[i]);
//                System.out.println(predictedDigit + "," + idealDigit);
                if(predictedDigit == idealDigit)
                    cnt++;
            }

            return 1 - (double)cnt / total;
        };
    }

    public static int toDigit(double[] arr) {
        double max = arr[0];
        int res = 0;

        for (int i = 0; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                res = i;
            }
        }

        return res;
    }
}
