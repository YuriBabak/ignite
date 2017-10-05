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

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteDataStreamer;
import org.apache.ignite.configuration.IgniteConfiguration;
import org.apache.ignite.internal.util.IgniteUtils;
import org.apache.ignite.ml.encog.caches.TestTrainingSetCache;
import org.apache.ignite.ml.encog.evolution.operators.IgniteEvolutionaryOperator;
import org.apache.ignite.ml.encog.evolution.operators.MutateNodes;
import org.apache.ignite.ml.encog.evolution.operators.NodeCrossover;
import org.apache.ignite.ml.encog.evolution.operators.WeightCrossover;
import org.apache.ignite.ml.encog.metaoptimizers.AddLeaders;
import org.apache.ignite.ml.encog.metaoptimizers.TopologyChanger;
import org.apache.ignite.ml.encog.util.Util;
import org.apache.ignite.ml.encog.wav.WavReader;
import org.apache.ignite.ml.math.functions.IgniteFunction;
import org.apache.ignite.testframework.junits.IgniteTestResources;
import org.apache.ignite.testframework.junits.common.GridCommonAbstractTest;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.TrainingSetScore;

/**
 * TODO: add description.
 */
public class WavTest extends GridCommonAbstractTest {
    private static final int NODE_COUNT = 3;
    private static String WAV_LOCAL = "/home/enny/Downloads/";

    /** Grid instance. */
    protected Ignite ignite;

    /**
     * Default constructor.
     */
    public WavTest() {
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

    //TODO: WIP
    public void test(){
        IgniteUtils.setCurrentIgniteName(ignite.configuration().getIgniteInstanceName());

        int framesInBatch = 1000;

        System.out.println("Reading wav...");
        List<double[]> rawData = WavReader.read(WAV_LOCAL + "sample4.wav", framesInBatch);
        System.out.println("Done.");

        int histDepth = 100;

        loadIntoCache(rawData, histDepth);

//
//        // create training data
        int n = 50;
        int k = 49;
//
        IgniteFunction<Integer, IgniteNetwork> fact = i -> {
            IgniteNetwork res = new IgniteNetwork();
            res.addLayer(new BasicLayer(null,false, histDepth));
            res.addLayer(new BasicLayer(new org.encog.engine.network.activation.ActivationSigmoid(),false,n));
            res.addLayer(new BasicLayer(new org.encog.engine.network.activation.ActivationSoftMax(),false,1));
            res.getStructure().finalizeStructure();

            res.reset();
            return res;
        };
//
        List<IgniteEvolutionaryOperator> evoOps = Arrays.asList(
            new NodeCrossover(0.2, "nc"),
            new WeightCrossover(0.2, "wc"),
//            new WeightMutation(0.2, 0.05, "wm"),
            new MutateNodes(10, 0.2, 0.05, "mn"));
//
        IgniteFunction<Integer, TopologyChanger.Topology> topologySupplier = (IgniteFunction<Integer, TopologyChanger.Topology>)subPop -> {
            Map<LockKey, Double> locks = new HashMap<>();

            int[] toDrop = Util.selectKDistinct(n, Math.abs(new Random().nextInt()) % k);

            for (int neuron : toDrop)
                locks.put(new LockKey(1, neuron), 0.0);

            return new TopologyChanger.Topology(locks);
        };
        int datasetSize = rawData.size() - histDepth - 1;
        System.out.println("DS size " + datasetSize);
        GaTrainerCacheInput input = new GaTrainerCacheInput<>(TestTrainingSetCache.NAME,
            fact,
            datasetSize,
            60,
            evoOps,
            30,
            (in, ignite) -> new TrainingSetScore(in.mlDataSet(ignite)),
            3,
            new TopologyChanger(topologySupplier).andThen(new AddLeaders(0.2))/*.andThen(new LearningRateAdjuster())*/,
            0.02
        );

        EncogMethodWrapper model = new GATrainer(ignite).train(input);
//
//        calculateError(model);
    }

    /**
     * Load wav into cache.
     *
     * @param wav Wav.
     * @param historyDepth History depth.
     */
    private void loadIntoCache(List<double[]> wav, int historyDepth) {
        TestTrainingSetCache.getOrCreate(ignite);

        try (IgniteDataStreamer<Integer, MLDataPair> stmr = ignite.dataStreamer(TestTrainingSetCache.NAME)) {
            // Stream entries.

            int samplesCnt = wav.size();
            System.out.println("Loading " + samplesCnt + " samples into cache...");
            for (int i = historyDepth; i < samplesCnt - 1; i++){

                BasicMLData dataSetEntry = new BasicMLData(wav.subList(i - historyDepth, i).stream().map(doubles ->
                    Arrays.stream(doubles).sum() / doubles.length).mapToDouble(d -> d).toArray());

                double[] rawLable = wav.get(i + 1);
                double[] lable = {Arrays.stream(rawLable).sum() / rawLable.length};

                stmr.addData(i - historyDepth, new BasicMLDataPair(dataSetEntry, new BasicMLData(lable)));
            }

        }
    }
}
