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

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.function.Consumer;
import org.apache.ignite.IgniteDataStreamer;
import org.apache.ignite.internal.util.IgniteUtils;
import org.apache.ignite.ml.Model;
import org.apache.ignite.ml.encog.caches.TestTrainingSetCache;
import org.apache.ignite.ml.encog.evolution.operators.CrossoverFeatures;
import org.apache.ignite.ml.encog.evolution.operators.IgniteEvolutionaryOperator;
import org.apache.ignite.ml.encog.evolution.operators.MutateNodes;
import org.apache.ignite.ml.encog.evolution.operators.NodeCrossover;
import org.apache.ignite.ml.encog.evolution.operators.WeightCrossover;
import org.apache.ignite.ml.encog.metaoptimizers.AddLeaders;
import org.apache.ignite.ml.encog.metaoptimizers.TopologyChanger;
import org.apache.ignite.ml.encog.util.Util;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.apache.ignite.ml.math.functions.IgniteFunction;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.TrainingSetScore;

/** IMPL NOTE do NOT run this as test class because JUnit3 will pick up redundant test from superclass. */
public class PredictionTracer extends GenTest {
    /** */
    private static final String MNIST_LOCATION = "C:/work/test/mnist/";

    /** */
    private static final String delim = ",";

    /** */
    public void testPrediction() throws IOException {
        IgniteUtils.setCurrentIgniteName(ignite.configuration().getIgniteInstanceName());

        System.out.println("Reading mnist...");
        MnistUtils.Pair<double[][], double[][]> mnist = MnistUtils.mnist(MNIST_LOCATION + "train-images-idx3-ubyte", MNIST_LOCATION + "train-labels-idx1-ubyte", new Random(), 60_000);
//        MnistUtils.Pair<double[][], double[][]> mnist = MnistUtils.mnist(MNIST_LOCATION + "t10k-images-idx3-ubyte", MNIST_LOCATION + "t10k-labels-idx1-ubyte", new Random(), 10_000);

        System.out.println("Done.");

        System.out.println("Loading MNIST into test cache...");
        loadIntoCache(mnist);
        System.out.println("Done.");

        // create training data
        int n = 50;
        int k = 149;

        IgniteFunction<Integer, IgniteNetwork> fact = i -> {
            IgniteNetwork res = new IgniteNetwork();
            res.addLayer(new BasicLayer(null,false,28 * 28));
            res.addLayer(new BasicLayer(new org.encog.engine.network.activation.ActivationSigmoid(),false,n));
            res.addLayer(new BasicLayer(new org.encog.engine.network.activation.ActivationSoftMax(),false,10));
            res.getStructure().finalizeStructure();

            res.reset();
            return res;
        };

        List<IgniteEvolutionaryOperator> evoOps = Arrays.asList(
            new NodeCrossover(0.2, "nc"),
            new WeightCrossover(0.2, "wc"),
            new CrossoverFeatures(0.1, "cf"),
//            new WeightMutation(0.2, 0.05, "wm"),
            new MutateNodes(10, 0.2, 0.1, "mn"));

        IgniteFunction<Integer, TopologyChanger.Topology> topSupplier = (IgniteFunction<Integer, TopologyChanger.Topology>)subPop -> {
            Map<LockKey, Double> locks = new HashMap<>();
            int toDropCnt = Math.abs(new Random().nextInt()) % k;

            int[] toDrop = Util.selectKDistinct(n, toDropCnt);

            for (int neuron : toDrop)
                locks.put(new LockKey(1, neuron), 0.0);

            System.out.println("For population " + subPop + " we dropped " + toDropCnt);

            return new TopologyChanger.Topology(locks);
        };
        GaTrainerCacheInput input = new GaTrainerCacheInput<>(TestTrainingSetCache.NAME,
            fact,
            mnist.getFst().length,
            60,
            evoOps,
            30,
            (in, ignite) -> new TrainingSetScore(in.mlDataSet(ignite)),
            3,
            new AddLeaders(0.2)
                .andThen(new TopologyChanger(topSupplier))
//                .andThen(new LearningRateAdjuster())
            ,
            0.02
        );

        @SuppressWarnings("unchecked")
        EncogMethodWrapper mdl = new GATrainer(ignite).train(input);

        calculateError(mdl);
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
        MnistUtils.Pair<double[][], double[][]> testMnistData = MnistUtils.mnist(
            MNIST_LOCATION + "t10k-images-idx3-ubyte", MNIST_LOCATION + "t10k-labels-idx1-ubyte",
            new Random(), 10_000);

        ResultsWriter writer = new ResultsWriter();

        writeHeader(writer);

        IgniteBiFunction<Model<MLData, double[]>, MnistUtils.Pair<double[][], double[][]>, Double> errorsPercentage
            = errorsPercentage(writer::append);

        Double accuracy = errorsPercentage.apply(mdl, testMnistData);

        System.out.println(">>> Errs percentage: " + accuracy);
        System.out.println(">>> Tracing data saved: " + writer);
    }

    /** */
    private void writeHeader(ResultsWriter writer) {
        writer.append("index" + delim + "expected" + delim + "actual");
    }

    /** */
    private IgniteBiFunction<Model<MLData, double[]>, MnistUtils.Pair<double[][],double[][]>, Double> errorsPercentage(
        Consumer<String> writer) {
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

                if(verifyPrediction(v[i], predict, i, writer))
                    cnt++;
            }

            return 1 - (double)cnt / total;
        };
    }

    /** */
    private boolean verifyPrediction(double[] exp, double[] predict, int idx, Consumer<String> writer) {
        int predictedDigit = toDigit(predict);
        int idealDigit = toDigit(exp);

        writer.accept(formatResults(idx, idealDigit, predictedDigit));

        return predictedDigit == idealDigit;
    }

    /** */
    private String formatResults(int idx, double exp, double actual) {
        assert !formatDouble(1000_000_001.1).contains(delim) : "Formatted results contain [" + delim + "].";

        return "" +
            idx +
            delim +
            formatDouble(exp) +
            delim +
            formatDouble(actual);
    }

    /** */
    private String formatDouble(double val) {
        return String.format(Locale.US, "%f", val);
    }

    /** */
    private static class ResultsWriter {
        /** */
        private final File tmp = createTmpFile();

        /** */
        void append(String res) {
            if (res == null)
                throw new IllegalArgumentException("Prediction tracing data is null.");

            try {
                writeResults(res, tmp);
            }
            catch (IOException e) {
                throw new RuntimeException("Failed to write to [" + this + "].");
            }
        }

        /** */
        private void writeResults(String res, File tmp) throws IOException {
            final String unixLineSeparator = "\n";

            try (final PrintWriter writer = new PrintWriter(Files.newBufferedWriter(Paths.get(tmp.toURI()),
                StandardOpenOption.APPEND, StandardOpenOption.CREATE))) {
                writer.write(res + unixLineSeparator);
            }
        }

        /** */
        private File createTmpFile() {
            final String prefix = UUID.randomUUID().toString(), suffix = ".csv";

            try {
                return File.createTempFile(prefix, suffix);
            }
            catch (IOException e) {
                throw new RuntimeException("Failed to create file [" + prefix + suffix + "].");
            }
        }

        /** {@inheritDoc} */
        @Override public String toString() {
            return "ResultsWriter{" + "tmp=" + tmp.getAbsolutePath() +
                '}';
        }
    }
}
