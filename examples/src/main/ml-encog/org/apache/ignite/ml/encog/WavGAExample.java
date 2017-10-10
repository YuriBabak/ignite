package org.apache.ignite.ml.encog;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
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
import org.apache.ignite.ml.encog.util.MSECalculator;
import org.apache.ignite.ml.encog.util.PredictedWavWriter;
import org.apache.ignite.ml.encog.util.SequentialRunner;
import org.apache.ignite.ml.encog.wav.WavReader;
import org.apache.ignite.ml.math.functions.IgniteFunction;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.TrainingSetScore;
import org.jetbrains.annotations.NotNull;

public class WavGAExample {
    private static final String OUTPUT_PRED_WAV_DEFAULT = "~/output.wav";
    private static int HISTORY_DEPTH_LOG_DEFAULT = 5;
    private static int MAX_TICKS_DEFAULT = 40;
    private static int FRAMES_IN_BATCH_DEFAULT = 2;
    private static int MAX_SAMPLES_DEFAULT = 1_000_000;

    public static void main(String[] args){

        String trainingSample = "~/wav/sample.4";
        String dataSample = "~/wav/sample.4";

        String igniteCfgPath = "examples/config/example-ml-nn-client.xml";

        CommandLineParser parser = new BasicParser();

        int histDepthLog;
        int maxSamples;
        int framesInBatch;
        int maxTicks;
        int histDepth;
        String outputWavPath;

        try {
            // parse the command line arguments
            CommandLine line = parser.parse( buildOptions(), args );

            histDepthLog = getIntOrDefault("depth_log", HISTORY_DEPTH_LOG_DEFAULT, line);
            maxSamples = getIntOrDefault("max_samples", MAX_SAMPLES_DEFAULT, line);
            framesInBatch = getIntOrDefault("fib", FRAMES_IN_BATCH_DEFAULT, line);
            maxTicks = getIntOrDefault("max_ticks", MAX_TICKS_DEFAULT, line);
            histDepth = (int)Math.pow(2, histDepthLog);
            outputWavPath = line.getOptionValue("out", OUTPUT_PRED_WAV_DEFAULT);

            trainingSample = line.getOptionValue("tr_sample");
            dataSample = line.getOptionValue("data_samples");

            if (line.hasOption("cfg")) {
                igniteCfgPath = line.getOptionValue("cfg");
                System.out.println("Starting with config " + igniteCfgPath);
            }

        }
        catch (ParseException e) {
            e.printStackTrace();
            return;
        }

        try (Ignite ignite = Ignition.start(igniteCfgPath)) {
            System.out.println("Reading wav...");

            WavReader.WavInfo inputWav = WavReader.read(trainingSample, framesInBatch);
            List<double[]> rawData = inputWav.batchs();
            System.out.println("Done.");

            long before = System.currentTimeMillis();
            CacheUtils.loadIntoCache(rawData, histDepth, maxSamples, CacheUtils.CACHE_NAME, ignite);

            IgniteFunction<Integer, IgniteNetwork> fact = getNNFactory(histDepthLog);

            double lr = 0.5;

            System.out.println("Learning rate is " + lr);

            List<IgniteEvolutionaryOperator> evoOps = Arrays.asList(
                new NodeCrossover(0.2, "nc"),
//              new CrossoverFeatures(0.2, "cf"),
//              new WeightCrossover(0.2, "wc"),
                new WeightMutation(0.2, lr, "wm"),
                new MutateNodes(10, 0.2, lr, "mn")
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
                4,
                new AddLeaders(0.2)
                    .andThen(new LearningRateAdjuster(null, 3))
                    .andThen(new BasicStatsCounter()),
                0.2,
                metaoptimizerData -> {
                    BasicStatsCounter.BasicStats stats = metaoptimizerData.get(0).get2();
                    int tick = stats.tick();
                    long msETA = stats.currentGlobalTickDuration() * (maxTicks - tick);
                    System.out.println("Current global iteration took " + stats.currentGlobalTickDuration() + "ms, ETA to end is " + (msETA / 1000 / 60) + "mins, " + (msETA / 1000 % 60) + " sec,");
                    return stats.tick() > maxTicks;
                }
            );

            EncogMethodWrapper mdl = new GATrainer<>(ignite).train(input);

            System.out.println("Training took " + (System.currentTimeMillis() - before) + " ms.");

            SequentialRunner runner = new SequentialRunner();

            runner.add(new MSECalculator());
            runner.add(new PredictedWavWriter(outputWavPath, inputWav.batchs().size(), (int)(inputWav.file().getSampleRate() / inputWav.file().getNumChannels())));

            runner.run(mdl, histDepth, framesInBatch, dataSample);
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

        Option histDepthOpt = OptionBuilder.withArgName("depth").withLongOpt("depth").hasArg()
            .withDescription("log base 2 of depth of history for prediction, default is " + HISTORY_DEPTH_LOG_DEFAULT)
            .isRequired(false).withType(Integer.TYPE).create();

        Option maxTicksOpt = OptionBuilder.withArgName("max_ticks").withLongOpt("max_ticks").hasArg()
            .withDescription("maximal count of global ticks " + MAX_TICKS_DEFAULT)
            .isRequired(false).withType(Integer.TYPE).create();

        Option framesInBatchOpt = OptionBuilder.withArgName("fib").withLongOpt("fib").hasArg()
            .withDescription("number of wav frames in batch, default is " + FRAMES_IN_BATCH_DEFAULT).isRequired(false)
            .withType(Integer.TYPE).create();

        Option trainingSamplesOpt = OptionBuilder.withArgName("tr_sample").withLongOpt("tr_sample").isRequired()
            .withDescription("path to sample").hasArgs().create();

        Option trainingDataSampleOpt = OptionBuilder.withArgName("data_samples").withLongOpt("data_samples").isRequired()
            .withDescription("path to data samples, uses for accuracy estimation").hasArg().create();

        Option maxSamplesOpt = OptionBuilder.withArgName("max_samples").withLongOpt("max_samples").hasArg()
            .withDescription("max count of samples, default is " + MAX_SAMPLES_DEFAULT).isRequired(false)
            .withType(Integer.TYPE).create();

        Option igniteConfOpt = OptionBuilder.withArgName("cfg").withLongOpt("cfg").hasArg().isRequired(false)
            .withDescription("path to ignite config, default is examples/config/example-ml-nn.xml").create();

        Option wavOutOpt = OptionBuilder.withArgName("out").withLongOpt("out").hasArg().isRequired(false)
            .withDescription("path to predicted wav, default is " + OUTPUT_PRED_WAV_DEFAULT).create();

        options.addOption(histDepthOpt);
        options.addOption(framesInBatchOpt);
        options.addOption(trainingSamplesOpt);
        options.addOption(trainingDataSampleOpt);
        options.addOption(igniteConfOpt);
        options.addOption(maxSamplesOpt);
        options.addOption(wavOutOpt);
        options.addOption(maxTicksOpt);

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
