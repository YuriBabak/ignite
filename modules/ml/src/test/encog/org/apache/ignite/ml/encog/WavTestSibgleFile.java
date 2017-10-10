package org.apache.ignite.ml.encog;/*
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

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteDataStreamer;
import org.apache.ignite.configuration.IgniteConfiguration;
import org.apache.ignite.internal.util.IgniteUtils;
import org.apache.ignite.ml.Model;
import org.apache.ignite.ml.encog.caches.TestTrainingSetCache;
import org.apache.ignite.ml.encog.caches.TestTrainingSetCacheSingleFileEntity;
import org.apache.ignite.ml.encog.evolution.operators.IgniteEvolutionaryOperator;
import org.apache.ignite.ml.encog.evolution.operators.MutateNodes;
import org.apache.ignite.ml.encog.evolution.operators.NodeCrossover;
import org.apache.ignite.ml.encog.evolution.operators.WeightMutation;
import org.apache.ignite.ml.encog.metaoptimizers.AddLeaders;
import org.apache.ignite.ml.encog.metaoptimizers.BasicStatsCounter;
import org.apache.ignite.ml.encog.metaoptimizers.LearningRateAdjuster;
import org.apache.ignite.ml.encog.viewbased.ViewGaTrainerCacheInput;
import org.apache.ignite.ml.encog.wav.WavReader;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.apache.ignite.ml.math.functions.IgniteFunction;
import org.apache.ignite.testframework.junits.IgniteTestResources;
import org.apache.ignite.testframework.junits.common.GridCommonAbstractTest;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.TrainingSetScore;

public class WavTestSibgleFile extends GridCommonAbstractTest {
    private static final int NODE_COUNT = 3;
    private static String WAV_LOCAL = "/home/enny/Downloads/wav/";

    /** Grid instance. */
    protected Ignite ignite;

    /**
     * Default constructor.
     */
    public WavTestSibgleFile() {
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

    /** {@inheritDoc} */
    @Override protected long getTestTimeout() {
        return 60000000;
    }

    /** {@inheritDoc} */
    @Override protected IgniteConfiguration getConfiguration(String igniteInstanceName,
        IgniteTestResources rsrcs) throws Exception {
        IgniteConfiguration configuration = super.getConfiguration(igniteInstanceName, rsrcs);
        configuration.setIncludeEventTypes();
        configuration.setPeerClassLoadingEnabled(true);
        configuration.setMetricsUpdateFrequency(2000);

        return configuration;
    }

    public void testSingleFile() throws IOException {
        IgniteUtils.setCurrentIgniteName(ignite.configuration().getIgniteInstanceName());

        int framesInBatch = 2;
        int sampleToRead = 4;
        int rate = 44100;

        System.out.println("Reading wav...");
        double[] rawData = WavReader.readAsSingleChannel(WAV_LOCAL + "sample" + sampleToRead + "_rate" + rate + "_channel1.wav");
        System.out.println("Done.");

        int pow = 9;
        int lookForwardFor = 1;
        int histDepth = (int)Math.pow(2, pow);

        int maxSamples = 1_000_000;
        loadIntoCacheAsSingleEntry(0, rawData);

        int n = 50;
        int k = 49;

        IgniteFunction<Integer, IgniteNetwork> fact = i -> buildTreeLikeNetComplex(pow, lookForwardFor);

        int datasetSize = rawData.length - histDepth + 1;

        double lr = 0.1;
        List<IgniteEvolutionaryOperator> evoOps = Arrays.asList(
            new NodeCrossover(0.2, "nc"),
            new WeightMutation(0.2, lr, "wm"),
            new MutateNodes(10, 0.2, lr, "mn")
        );

        System.out.println("DS size " + datasetSize);
        Integer maxTicks = 40;

        ViewGaTrainerCacheInput input = new ViewGaTrainerCacheInput<>(TestTrainingSetCache.NAME,
            fact,
            datasetSize,
            60,
            evoOps,
            30,
            // TODO: for the moment each population gets the same dataset, can be tweaked
            (in, ignite) -> new TrainingSetScore(in.mlDataSet(0, ignite)),
            3,
            new AddLeaders(0.2)
                .andThen(new LearningRateAdjuster(null, 3))
                .andThen(new BasicStatsCounter())
            ,
            0.00001,
            metaoptimizerData -> {
                BasicStatsCounter.BasicStats stats = metaoptimizerData.get(0).get2();
                int tick = stats.tick();
                long msETA = stats.currentGlobalTickDuration() * (maxTicks - tick);
                System.out.println("Current global iteration took " + stats.currentGlobalTickDuration() + "ms, ETA to end is " + (msETA / 1000 / 60) + "mins, " + (msETA / 1000 % 60) + " sec,");
                return stats.tick() > maxTicks;
            },
            histDepth
        );

        EncogMethodWrapper model = new GATrainer(ignite).train(input);

        calculateError(model, rate, sampleToRead, histDepth, framesInBatch);
    }

    private void loadIntoCacheAsSingleEntry(int sample, double[] wav) {
        TestTrainingSetCacheSingleFileEntity.getOrCreate(ignite);

        try (IgniteDataStreamer<Integer, double[]> stmr = ignite.dataStreamer(TestTrainingSetCacheSingleFileEntity.NAME)) {
            // Stream entries.
            stmr.addData(sample, wav);
        }
    }

    private void calculateError(EncogMethodWrapper model, int rate, int sampleNumber, int historyDepth, int framesInBatch) throws IOException {
        List<double[]> rawData = WavReader.read(WAV_LOCAL + "sample" + sampleNumber + "_rate" + rate + ".wav", framesInBatch);

        IgniteBiFunction<Model<MLData, double[]>, List<double[]>, Double> errorsPercentage = errorsPercentage(sampleNumber, rate, historyDepth);
        Double accuracy = errorsPercentage.apply(model, rawData);
        System.out.println(">>> Errs estimation: " + accuracy);
    }

    private static IgniteNetwork buildTreeLikeNetComplex(int leavesCountLog, int lookForwardCnt) {
        IgniteNetwork res = new IgniteNetwork();

        int lastTreeLike = 0;
        for (int i = leavesCountLog; i >=0 && Math.pow(2, i) >= (lookForwardCnt / 2); i--) {
            res.addLayer(new BasicLayer(i == 0 ? null : new ActivationSigmoid(), false, (int)Math.pow(2, i)));
            lastTreeLike = leavesCountLog - i;
        }

        res.addLayer(new BasicLayer(new ActivationSigmoid(), false, lookForwardCnt));

        res.getStructure().finalizeStructure();

        for (int i = 0; i < lastTreeLike; i++) {
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

    private IgniteBiFunction<Model<MLData, double[]>, List<double[]>, Double> errorsPercentage(int sample, int rate, int historyDepth){

        return (model, ds) -> {
            double buff[] = new double[ds.size() - historyDepth];
            double genBuff[] = new double[ds.size()];

            double cnt = 0L;

            int samplesCnt = ds.size();

            // Sample for generation taken from middle
            List<Double> inputForGen = ds.subList(samplesCnt / 2, samplesCnt / 2 + historyDepth).stream().map(doubles ->
                (Arrays.stream(doubles).sum() / doubles.length + 1) / 2).collect(Collectors.toList());

            System.arraycopy(inputForGen.stream().mapToDouble(d -> d).toArray(), 0, genBuff, 0, historyDepth);

            for (int i = historyDepth; i < samplesCnt - 1; i++){

                BasicMLData dataSetEntry = new BasicMLData(ds.subList(i - historyDepth, i).stream().map(doubles ->
                    (Arrays.stream(doubles).sum() / doubles.length + 1) / 2).mapToDouble(d -> d).toArray());

                BasicMLData genDataSetEntry = new BasicMLData(inputForGen.stream().mapToDouble(d -> d).toArray());

                double[] rawLable = ds.get(i + 1);
                double[] lable = {(Arrays.stream(rawLable).sum() / rawLable.length + 1) / 2};

                double[] predict = model.predict(dataSetEntry);

                buff[i - historyDepth] = predict[0] - 1;
                double genPred = model.predict(genDataSetEntry)[0] - 1;
                genBuff[i] = genPred;
                inputForGen.remove(0);
                inputForGen.add(historyDepth - 1, genPred);

                cnt += predict[0] * predict[0] - lable[0] * lable[0];
            }

            WavReader.write(WAV_LOCAL + "sample" + sample + "_rate" + rate + "_3.wav", buff, rate);
            WavReader.write(WAV_LOCAL + "sample" + sample + "_rate" + rate + "_gen.wav", genBuff, rate);

            return cnt / samplesCnt;
        };
    }
}
