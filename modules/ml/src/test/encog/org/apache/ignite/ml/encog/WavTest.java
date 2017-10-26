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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.ignite.internal.util.IgniteUtils;
import org.apache.ignite.ml.encog.caches.TestTrainingSetCache;
import org.apache.ignite.ml.encog.evolution.operators.IgniteEvolutionaryOperator;
import org.apache.ignite.ml.encog.evolution.operators.MutateNodes;
import org.apache.ignite.ml.encog.evolution.operators.NodeCrossover;
import org.apache.ignite.ml.encog.evolution.operators.WeightMutation;
import org.apache.ignite.ml.encog.metaoptimizers.AddLeaders;
import org.apache.ignite.ml.encog.metaoptimizers.BasicStatsCounter;
import org.apache.ignite.ml.encog.metaoptimizers.LearningRateAdjuster;
import org.apache.ignite.ml.encog.metaoptimizers.TopologyChanger;
import org.apache.ignite.ml.encog.util.GeneratedWavWriter;
import org.apache.ignite.ml.encog.util.MSECalculator;
import org.apache.ignite.ml.encog.util.PredictedWavWriter;
import org.apache.ignite.ml.encog.util.SequentialRunner;
import org.apache.ignite.ml.encog.util.Util;
import org.apache.ignite.ml.encog.wav.WavReader;
import org.apache.ignite.ml.math.functions.IgniteFunction;
import org.encog.neural.networks.BasicNetwork;

import static org.apache.ignite.ml.encog.NeuralNetworkUtils.buildTreeLikeNetComplex;

/**
 * TODO: add description.
 */
public class WavTest extends AbstractWavTest {
    private static final int NODE_COUNT = 3;
    private static String WAV_LOCAL = "/home/enny/Downloads/wav/";

    @Override protected String getWavLocalPath() {
        return WAV_LOCAL;
    }

    @Override protected int getNodeCount() {
        return NODE_COUNT;
    }

    /**
     * Default constructor.
     */
    public WavTest() {
        super();
    }

//    //TODO: WIP
    public void test() throws IOException {
        IgniteUtils.setCurrentIgniteName(ignite.configuration().getIgniteInstanceName());

        int framesInBatch = 2;
        int sampleToRead = 4;
        int rate = 44100;
        int subPopsCount = 2;

        System.out.println("Reading wav...");
        String dataSample = WAV_LOCAL + "sample" + sampleToRead + "_rate" + rate + ".wav";
        WavReader.WavInfo inputWav = WavReader.read(dataSample, framesInBatch);
        List<double[]> rawData = inputWav.batchs();
        System.out.println("Done.");

        int pow = 7;
        int lookForwardFor = 1;
        int histDepth = (int)Math.pow(2, pow);

        int maxSamples = 1_000_000;
        loadIntoCache(rawData, histDepth, maxSamples, lookForwardFor);

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
            return buildTreeLikeNetComplex(pow - (i % subPopsCount), lookForwardFor);
        };

        IgniteFunction<Integer, TreeNetwork> fact1 = i -> new TreeNetwork(pow + 1);
//
        double lr = 0.1;
        List<IgniteEvolutionaryOperator> evoOps = Arrays.asList(
            new NodeCrossover(0.2, "nc"),
//            new CrossoverFeatures(0.2, "cf"),
//            new WeightCrossover(0.2, "wc"),
            new WeightMutation(0.2, lr, "wm"),
            new MutateNodes(0.1, 0.2, lr, "mn")
        );
//
        IgniteFunction<Integer, TopologyChanger.Topology> topologySupplier = (IgniteFunction<Integer, TopologyChanger.Topology>)subPop -> {
            Map<LockKey, Double> locks = new HashMap<>();
            int toDropCnt = Math.abs(new Random().nextInt()) % k;

            int[] toDrop = Util.selectKDistinct(n, Math.abs(new Random().nextInt()) % k);

            for (int neuron : toDrop)
                locks.put(new LockKey(1, neuron), 0.0);

            System.out.println("For population " + subPop + " we dropped " + toDropCnt);

            return new TopologyChanger.Topology(locks);
        };
        int datasetSize = Math.min(maxSamples, rawData.size() - histDepth - 1);
        System.out.println("DS size " + datasetSize);
        Integer maxTicks = 10;

        GaTrainerCacheInput input = new GaTrainerCacheInput<>(TestTrainingSetCache.NAME,
            fact,
            datasetSize,
            60,
            evoOps,
            sp -> 30 * (int)Math.pow(2, sp), // tree depth drops with grow of subpopulation number, so we can do twice more local ticks with each level drop.
            // TODO: for the moment each population gets the same dataset, can be tweaked
            (in, ignite) -> new AdaptableTrainingScore(in.mlDataSet(0, ignite)),
            subPopsCount,
            new AddLeaders(0.2)
                .andThen(new LearningRateAdjuster(null, subPopsCount))
                .andThen(new BasicStatsCounter())
            /*.andThen(new LearningRateAdjuster())*/
            ,
            0.02,
            metaoptimizerData -> {
                BasicStatsCounter.BasicStats stats = metaoptimizerData.get(0).get2();
                int tick = stats.tick();
                long msETA = stats.currentGlobalTickDuration() * (maxTicks - tick);
                System.out.println("Current global iteration took " + stats.currentGlobalTickDuration() + "ms, ETA to end is " + (msETA / 1000 / 60) + "mins, " + (msETA / 1000 % 60) + " sec,");
                return stats.tick() > maxTicks;
            }
        );



        EncogMethodWrapper model = new GATrainer(ignite).train(input);

        System.out.println("Best history depth " + ((BasicNetwork)model.getM()).getLayerNeuronCount(0));

//        PersistorRegistry.getInstance().add(new PersistIgniteNetwork());
//        EncogDirectoryPersistence.saveObject(new File(WAV_LOCAL + "net_" + sampleToRead + ".nn"), model.getM());
//
//        calculateError(model, rate, sampleToRead, ((BasicNetwork)model.getM()).getLayerNeuronCount(0), framesInBatch);

        SequentialRunner runner = new SequentialRunner();

        String outputWavPath = WAV_LOCAL + "sample" + sampleToRead + "_rate" + rate + "_pred.wav";
        int size = inputWav.batchs().size();
        int r = (int)(inputWav.file().getSampleRate() / inputWav.file().getNumChannels());

        int stepSize = 1;
        double csvDown = 1.0;

        runner.add(new MSECalculator());
        runner.add(new PredictedWavWriter(outputWavPath, size, r, lookForwardFor));
//        runner.add(new WavTracer((int)(stepSize * csvDown), 1_000_000));
        String outputGenWavPath = WAV_LOCAL + "sample" + sampleToRead + "_rate" + rate + "_gen.wav";

        if (outputGenWavPath != null) {
            System.out.println("Added generating in " + outputGenWavPath);
            runner.add(new GeneratedWavWriter(model.getM(), outputGenWavPath, size, r));
        }

        int bestMdlHistDepth = ((BasicNetwork)model.getM()).getLayerNeuronCount(0);

        System.out.println("Best model history depth: " + bestMdlHistDepth);
        runner.run(model, bestMdlHistDepth, framesInBatch, dataSample);


//        System.out.println(NeuralNetworkUtils.printBinaryNetwork((BasicNetwork)model.getM()));
    }
}
