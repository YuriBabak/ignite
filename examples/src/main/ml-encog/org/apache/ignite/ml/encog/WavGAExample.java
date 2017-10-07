package org.apache.ignite.ml.encog;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.ignite.Ignite;
import org.apache.ignite.Ignition;
import org.apache.ignite.ml.encog.evolution.operators.IgniteEvolutionaryOperator;
import org.apache.ignite.ml.encog.evolution.operators.MutateNodes;
import org.apache.ignite.ml.encog.evolution.operators.NodeCrossover;
import org.apache.ignite.ml.encog.evolution.operators.WeightMutation;
import org.apache.ignite.ml.encog.metaoptimizers.AddLeaders;
import org.apache.ignite.ml.encog.metaoptimizers.LearningRateAdjuster;
import org.apache.ignite.ml.encog.wav.WavReader;
import org.apache.ignite.ml.math.functions.IgniteFunction;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.TrainingSetScore;
import org.jetbrains.annotations.NotNull;

public class WavGAExample {

    public static void main(String[] args){
        int histDepth = 240;
        String trainingSample = "~/wav/sample.4";
        int framesInBatch = 50;
        String dataSample = "~/wav/sample.4";

        CommandLineParser parser = new DefaultParser();

        try {
            // parse the command line arguments
            CommandLine line = parser.parse( buildOptions(), args );

            if (line.hasOption("depth"))
                histDepth = Integer.parseInt(line.getOptionValue("depth"));
            if (line.hasOption("fib"))
                framesInBatch = Integer.parseInt(line.getOptionValue("fib"));

            trainingSample = line.getOptionValue("tr_samples");
            dataSample = line.getOptionValue("data_samples");

        }
        catch (ParseException e) {
            e.printStackTrace();
            return;
        }

        Estimator estimator = new Estimator();

        try (Ignite ignite = Ignition.start("examples/config/example-ml-nn.xml")) {


            System.out.println("Reading wav...");
            List<double[]> rawData = WavReader.read(trainingSample, framesInBatch);
            System.out.println("Done.");



            int maxSamples = Integer.MAX_VALUE;
            CacheUtils.loadIntoCache(rawData, histDepth, maxSamples, CacheUtils.CACHE_NAME, ignite);

            IgniteFunction<Integer, IgniteNetwork> fact = getNNFactory(histDepth);

            List<IgniteEvolutionaryOperator> evoOps = Arrays.asList(
                new NodeCrossover(0.2, "nc"),
//              new CrossoverFeatures(0.2, "cf"),
//              new WeightCrossover(0.2, "wc"),
                new WeightMutation(0.2, 0.05, "wm"),
                new MutateNodes(10, 0.2, 0.05, "mn")
            );

            int datasetSize = Math.min(maxSamples, rawData.size() - histDepth - 1);
            System.out.println("DS size " + datasetSize);
            GaTrainerCacheInput input = new GaTrainerCacheInput<>(CacheUtils.CACHE_NAME,
                fact,
                datasetSize,
                60,
                evoOps,
                30,
                (in, ign) -> new TrainingSetScore(in.mlDataSet(ign)),
                3,
                new AddLeaders(0.2).andThen(new LearningRateAdjuster()),
                0.2
            );

            EncogMethodWrapper mdl = new GATrainer(ignite).train(input);

            estimator.calculateError(mdl, histDepth, framesInBatch, dataSample);
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Build cli options.
     */
    @NotNull private static Options buildOptions() {
        Options options = new Options();

        Option.Builder builder = Option.builder();

        Option histDepthOpt = builder.argName("depth").longOpt("depth").hasArg()
            .desc("depth of history for prediction, default is 240").optionalArg(true).type(Integer.TYPE).build();
        Option framesInBatchOpt = builder.argName("fib").longOpt("fib").hasArg()
            .desc("number of wav frames in batch, default is 50").optionalArg(true).type(Integer.TYPE).build();

        Option trainingSamplesOpt = builder.argName("tr_samples").longOpt("tr_samples").required()
            .desc("path to sample").hasArgs().build();
        Option trainingDataSampleOpt = builder.argName("data_samples").longOpt("data_samples").required()
            .desc("path to data samples, uses for accuracy estimation").hasArg().build();

        options.addOption(histDepthOpt);
        options.addOption(framesInBatchOpt);
        options.addOption(trainingSamplesOpt);
        options.addOption(trainingDataSampleOpt);

        return options;
    }

    /** */
    @NotNull private static IgniteFunction<Integer, IgniteNetwork> getNNFactory(int leavesCountLog) {
        return subPopulation -> {
            IgniteNetwork res = new IgniteNetwork();
            for (int i = leavesCountLog; i >=0; i--)
                res.addLayer(new BasicLayer(i == 0 ? null : new ActivationSigmoid(), false, (int)Math.pow(2, i)));

            res.getStructure().finalizeStructure();

            for (int i = 0; i < leavesCountLog - 1; i++) {
                for (int n = 0; n < res.getLayerNeuronCount(i); n += 2) {
                    res.dropOutputsFrom(i, n);
                    res.dropOutputsFrom(i, n + 1);

                    res.enableConnection(i, n, n / 2, true);
                    res.enableConnection(i, n + 1, n / 2, true);
                }
            }

            res.reset();

            return res;
        };
    }
}
