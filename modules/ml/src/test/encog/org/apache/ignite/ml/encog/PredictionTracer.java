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
import org.apache.ignite.ml.encog.evolution.operators.IgniteEvolutionaryOperator;
import org.apache.ignite.ml.encog.evolution.operators.MutateNodes;
import org.apache.ignite.ml.encog.evolution.operators.NodeCrossover;
import org.apache.ignite.ml.encog.evolution.operators.WeightMutation;
import org.apache.ignite.ml.encog.metaoptimizers.AddLeaders;
import org.apache.ignite.ml.encog.metaoptimizers.BasicStatsCounter;
import org.apache.ignite.ml.encog.metaoptimizers.LearningRateAdjuster;
import org.apache.ignite.ml.encog.metaoptimizers.TopologyChanger;
import org.apache.ignite.ml.encog.util.Util;
import org.apache.ignite.ml.encog.wav.WavReader;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.apache.ignite.ml.math.functions.IgniteFunction;
import org.encog.engine.network.activation.ActivationSigmoid;
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

        int framesInBatch = 2;

        System.out.println("Reading wav...");
        List<double[]> rawData = WavReader.read(WAV_LOCAL + "sample4.wav", framesInBatch);
        System.out.println("Done.");

        int pow = 4;
        int histDepth = (int)Math.pow(2, pow);

        int maxSamples = 2_000_000;
        loadIntoCache(rawData, histDepth, maxSamples);

        int n = 50;
        int k = 49;

        IgniteFunction<Integer, IgniteNetwork> fact = i -> {
//            IgniteNetwork res = new IgniteNetwork();
//            res.addLayer(new BasicLayer(null,false, histDepth));
//            res.addLayer(new BasicLayer(new org.encog.engine.network.activation.ActivationSigmoid(),false,n));
//            res.addLayer(new BasicLayer(new org.encog.engine.network.activation.ActivationSigmoid(),false,n));
//            res.addLayer(new BasicLayer(new org.encog.engine.network.activation.ActivationSigmoid(),false,n));
//            res.addLayer(new BasicLayer(new org.encog.engine.network.activation.ActivationSoftMax(),false,1));
//            res.getStructure().finalizeStructure();
//
//
//            res.reset();
            return buildTreeLikeNet(pow);
        };
//
        double lr = 0.5;
        List<IgniteEvolutionaryOperator> evoOps = Arrays.asList(
            new NodeCrossover(0.2, "nc"),
//            new CrossoverFeatures(0.2, "cf"),
//            new WeightCrossover(0.2, "wc"),
            new WeightMutation(0.2, lr, "wm"),
            new MutateNodes(10, 0.2, lr, "mn")
        );
//
        IgniteFunction<Integer, TopologyChanger.Topology> topSupplier =
            (IgniteFunction<Integer, TopologyChanger.Topology>)subPop -> {
            Map<LockKey, Double> locks = new HashMap<>();
            int toDropCnt = Math.abs(new Random().nextInt()) % k;

            int[] toDrop = Util.selectKDistinct(n, Math.abs(new Random().nextInt()) % k);

            for (int neuron : toDrop)
                locks.put(new LockKey(1, neuron), 0.0);

            System.out.println("For population " + subPop + " we dropped " + toDropCnt);

            return new TopologyChanger.Topology(locks);
        };
        int maxTicks = 40;
        int datasetSize = Math.min(maxSamples, rawData.size() - histDepth - 1);
        System.out.println("DS size " + datasetSize);
        GaTrainerCacheInput input = new GaTrainerCacheInput<>(TestTrainingSetCache.NAME,
            fact,
            datasetSize,
            60,
            evoOps,
            30,
            (in, ignite) -> new TrainingSetScore(in.mlDataSet(ignite)),
            3,
            new TopologyChanger(topSupplier)
                .andThen(new AddLeaders(0.2))
                .andThen(new BasicStatsCounter())/*.andThen(new LearningRateAdjuster())*/,
            0.02,
            map -> map.get(0).get2().tick() > maxTicks
            new AddLeaders(0.2).andThen(new LearningRateAdjuster())/*.andThen(new LearningRateAdjuster())*/,
            0.02
        );

        @SuppressWarnings("unchecked")
        EncogMethodWrapper mdl = new GATrainer(ignite).train(input);

        calculateError(mdl, histDepth);
    }

    /**
     * Load wav into cache.
     *
     * @param wav Wav.
     * @param histDepth History depth.
     */
    private void loadIntoCache(List<double[]> wav, int histDepth, int maxSamples) {
        TestTrainingSetCache.getOrCreate(ignite);

        try (IgniteDataStreamer<Integer, MLDataPair> stmr = ignite.dataStreamer(TestTrainingSetCache.NAME)) {
            // Stream entries.

            int samplesCnt = wav.size();
            System.out.println("Loading " + samplesCnt + " samples into cache...");
            for (int i = histDepth; i < samplesCnt - 1 && (i - histDepth) < maxSamples; i++){

                // The mean is calculated inefficient
                BasicMLData dataSetEntry = new BasicMLData(wav.subList(i - histDepth, i).stream().map(doubles ->
                    (Arrays.stream(doubles).sum() / doubles.length + 1) / 2).mapToDouble(d -> d).toArray());

                double[] rawLb = wav.get(i + 1);
                double[] lb = {(Arrays.stream(rawLb).sum() / rawLb.length + 1) / 2};

                stmr.addData(i - histDepth, new BasicMLDataPair(dataSetEntry, new BasicMLData(lb)));

                if (i % 5000 == 0)
                    System.out.println("Loaded " + i);
            }
            System.out.println("Done");
        }
    }

    /** */
    private void calculateError(EncogMethodWrapper mdl, int histDepth) throws IOException {
        List<double[]> rawData = WavReader.read(WAV_LOCAL + "sample4.wav", 100);

        ResultsWriter writer = new ResultsWriter();

        writeHeader(writer);

        IgniteBiFunction<Model<MLData, double[]>, List<double[]>, Double> errorsPercentage = errorsPercentage(histDepth,
            writer::append);
        Double accuracy = errorsPercentage.apply(mdl, rawData);
        System.out.println(">>> Errs estimation: " + accuracy);
        System.out.println(">>> Tracing data saved: " + writer);
    }

    /** */
    private static IgniteNetwork buildTreeLikeNet(int leavesCntLog) {

        IgniteNetwork res = new IgniteNetwork();
        for (int i = leavesCntLog; i >=0; i--)
            res.addLayer(new BasicLayer(i == 0 ? null : new ActivationSigmoid(), false, (int)Math.pow(2, i)));

        res.getStructure().finalizeStructure();

        for (int i = 0; i < leavesCntLog - 1; i++) {
            for (int n = 0; n < res.getLayerNeuronCount(i); n += 2) {
                res.dropOutputsFrom(i, n);
                res.dropOutputsFrom(i, n + 1);

                res.enableConnection(i, n, n / 2, true);
                res.enableConnection(i, n + 1, n / 2, true);
            }
        }

        res.reset();

        return res;
    }

    /** */
    private IgniteBiFunction<Model<MLData, double[]>, List<double[]>, Double> errorsPercentage(int histDepth,
        Consumer<String> writer){
        return (model, ds) -> {
            double cnt = 0L;

            int samplesCnt = ds.size();

            for (int i = histDepth; i < samplesCnt-1; i++){

                BasicMLData dataSetEntry = new BasicMLData(ds.subList(i - histDepth, i).stream().map(doubles ->
                    (Arrays.stream(doubles).sum() / doubles.length + 1) / 2).mapToDouble(d -> d).toArray());

                double[] rawLb = ds.get(i + 1);
                double[] lb = {(Arrays.stream(rawLb).sum() / rawLb.length + 1) / 2};

                double[] predict = model.predict(dataSetEntry);

                cnt += verifyPrediction(lb[0], predict[0], i, writer);
            }

            return cnt / samplesCnt;
        };
    }

    /** */
    private void writeHeader(ResultsWriter writer) {
        writer.append("index" + delim + "expected" + delim + "actual");
    }

    /** */
    private double verifyPrediction(double exp, double predict, int idx, Consumer<String> writer) {
        writer.accept(formatResults(idx, exp, predict));

        return predict * predict - exp * exp;
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
