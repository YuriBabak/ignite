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
import org.apache.ignite.ml.encog.metaoptimizers.BasicStatsCounter;
import org.apache.ignite.ml.encog.metaoptimizers.LearningRateAdjuster;
import org.apache.ignite.ml.encog.wav.WavReader;
import org.apache.ignite.ml.math.functions.IgniteFunction;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.TrainingSetScore;
import org.jetbrains.annotations.NotNull;

public class WavGAExample {
    private static int HISTORY_DEPTH_LOG_DEFAULT = 6;
    private static int MAX_TICKS_DEFAULT = 40;
    private static int FRAMES_IN_BATCH_DEFAULT = 2;
    private static int MAX_SAMPLES_DEFAULT = 1_000_000;

    public static void main(String[] args){

        String trainingSample = "~/wav/sample.4";
        String dataSample = "~/wav/sample.4";

        String igniteConfigPath = "examples/config/example-ml-nn.xml";

        CommandLineParser parser = new DefaultParser();

        int histDepthLog;
        int maxSamples;
        int framesInBatch;
        int maxTicks;
        int histDepth;

        try {
            // parse the command line arguments
            CommandLine line = parser.parse( buildOptions(), args );

            histDepthLog = getIntOrDefault("depth_log", HISTORY_DEPTH_LOG_DEFAULT, line);
            maxSamples = getIntOrDefault("max_samples", MAX_SAMPLES_DEFAULT, line);
            framesInBatch = getIntOrDefault("fib", FRAMES_IN_BATCH_DEFAULT, line);
            maxTicks = getIntOrDefault("max_ticks", MAX_TICKS_DEFAULT, line);
            histDepth = (int)Math.pow(2, histDepthLog);

            trainingSample = line.getOptionValue("tr_samples");
            dataSample = line.getOptionValue("data_samples");

            if (line.hasOption("cfg"))
                igniteConfigPath = line.getOptionValue("cfg");

        }
        catch (ParseException e) {
            e.printStackTrace();
            return;
        }

        Estimator estimator = new Estimator();

        try (Ignite ignite = Ignition.start(igniteConfigPath)) {
            System.out.println("Reading wav...");
            List<double[]> rawData = WavReader.read(trainingSample, framesInBatch);
            System.out.println("Done.");

            CacheUtils.loadIntoCache(rawData, histDepth, maxSamples, CacheUtils.CACHE_NAME, ignite);

            IgniteFunction<Integer, IgniteNetwork> fact = getNNFactory(histDepthLog);

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
                (in, ign) -> new TrainingSetScore(in.mlDataSet(0, ign)),
                3,
                new AddLeaders(0.2)
                    .andThen(new LearningRateAdjuster(null, 3))
                    .andThen(new BasicStatsCounter()),
                0.2,
                m -> m.get(0).get2().tick() > maxTicks
            );

            EncogMethodWrapper mdl = new GATrainer(ignite).train(input);

            estimator.calculateError(mdl, histDepth, framesInBatch, dataSample);
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static int getIntOrDefault(String optionName, int def, CommandLine line) {
        return line.hasOption(optionName) ? Integer.parseInt(line.getOptionValue(optionName)) : def;
    }

    /**
     * Build cli options.
     */
    @NotNull private static Options buildOptions() {
        Options options = new Options();

        Option.Builder builder = Option.builder();

        Option histDepthOpt = builder.argName("depth").longOpt("depth").hasArg()
            .desc("log base 2 of depth of history for prediction, default is " + HISTORY_DEPTH_LOG_DEFAULT).required(false).type(Integer.TYPE).build();
        Option framesInBatchOpt = builder.argName("fib").longOpt("fib").hasArg()
            .desc("number of wav frames in batch, default is " + FRAMES_IN_BATCH_DEFAULT).required(false).type(Integer.TYPE).build();
        Option trainingSamplesOpt = builder.argName("tr_samples").longOpt("tr_samples").required()
            .desc("path to sample").hasArgs().build();
        Option trainingDataSampleOpt = builder.argName("data_samples").longOpt("data_samples").required()
            .desc("path to data samples, uses for accuracy estimation").hasArg().build();
        Option maxSamplesOpt = builder.argName("max_samples").longOpt("max_samples").hasArg()
            .desc("max count of samples, default is " + MAX_SAMPLES_DEFAULT).required(false).type(Integer.TYPE).build();

        Option igniteConfOpt = builder.argName("cfg").longOpt("cfg").required(false)
            .desc("path to ignite config, default is examples/config/example-ml-nn.xml").build();

        options.addOption(histDepthOpt);
        options.addOption(framesInBatchOpt);
        options.addOption(trainingSamplesOpt);
        options.addOption(trainingDataSampleOpt);
        options.addOption(igniteConfOpt);
        options.addOption(maxSamplesOpt);

        return options;
    }

    /** */
    @NotNull private static IgniteFunction<Integer, IgniteNetwork> getNNFactory(int leavesCountLog) {
        return subPopulation -> {
            IgniteNetwork res = new IgniteNetwork();
            for (int i = leavesCountLog; i >= 0; i--)
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
