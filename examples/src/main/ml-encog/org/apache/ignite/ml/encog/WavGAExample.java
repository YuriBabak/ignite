package org.apache.ignite.ml.encog;

import java.io.File;
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
import org.apache.ignite.ml.encog.util.GeneratedWavWriter;
import org.apache.ignite.ml.encog.util.MSECalculator;
import org.apache.ignite.ml.encog.util.PredictedWavWriter;
import org.apache.ignite.ml.encog.util.SequentialRunner;
import org.apache.ignite.ml.encog.util.WavTracer;
import org.apache.ignite.ml.encog.wav.WavReader;
import org.apache.ignite.ml.math.functions.IgniteFunction;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.persist.PersistorRegistry;
import org.jetbrains.annotations.NotNull;

public class WavGAExample {
    private static final String OUTPUT_PRED_WAV_DEFAULT = "~/output.wav";
    private static final double CSV_DOWNSAMPLE_RATIO = 0.5;
    private static final int FORWARD_DEPTH = 1;
    private static final String DEFAULT_MODEL_PATH = "/home/artemmalykh/";
    private static double DEFAULT_BATCH_PERCENTAGE = 0.2;
    private static int HISTORY_DEPTH_LOG_DEFAULT = 5;
    private static int MAX_TICKS_DEFAULT = 40;
    private static int FRAMES_IN_BATCH_DEFAULT = 2;
    private static int MAX_SAMPLES_DEFAULT = 200_000;
    private static int SUBPOPS_DEFAULT = 3;
    private static int STEP_SIZE_DEFAULT = 1;

    public static void main(String[] args) {
        String trainingSample = "~/wav/sample.4";
        String dataSample = "~/wav/sample.4";

        String igniteCfgPath = "examples/config/example-ml-nn-client.xml";

        CommandLineParser parser = new BasicParser();

        int histDepthLog;
        int maxSamples;
        int framesInBatch;
        int maxTicks;
        int histDepth;
        int subpops;
        int forwardDepth;
        String outputWavPath;
        String outputGenWavPath;
        int stepSize;
        double batchPercentage;
        double csvDown;
        String modelsPath;

        try {
//            -tr_sample /home/artemmalykh/wavs/sample4_rate44100.wav -out /home/artemmalykh/out.wav -data_samples /home/artemmalykh/wavs/sample4_rate44100.wav -cfg /home/artemmalykh/example-ml-nn-client.xml -max_ticks 25 -depth_log 8 -out_gen /home/artemmalykh/out-gen.wav -subpops 3 -ds_part 0.05 -step_size 2 -fwd 2

            // parse the command line arguments
            CommandLine line = parser.parse( buildOptions(), args );

            histDepthLog = getIntOrDefault("depth_log", HISTORY_DEPTH_LOG_DEFAULT, line);
            maxSamples = getIntOrDefault("max_samples", MAX_SAMPLES_DEFAULT, line);
            framesInBatch = getIntOrDefault("fib", FRAMES_IN_BATCH_DEFAULT, line);
            maxTicks = getIntOrDefault("max_ticks", MAX_TICKS_DEFAULT, line);
            histDepth = (int)Math.pow(2, histDepthLog);
            outputWavPath = line.getOptionValue("out", OUTPUT_PRED_WAV_DEFAULT);
            outputGenWavPath = line.getOptionValue("out_gen");
            subpops = getIntOrDefault("subpops", SUBPOPS_DEFAULT, line);
            batchPercentage = getDoubleOrDefault("ds_part", DEFAULT_BATCH_PERCENTAGE, line);
            stepSize = getIntOrDefault("step_size", STEP_SIZE_DEFAULT, line);
            csvDown = getDoubleOrDefault("cdr", CSV_DOWNSAMPLE_RATIO, line);
            forwardDepth = getIntOrDefault("fwd", FORWARD_DEPTH, line);
            modelsPath = line.getOptionValue("mdl", DEFAULT_MODEL_PATH);

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
            SamplesCache.loadIntoCache(rawData, histDepth, maxSamples, stepSize, forwardDepth, ignite);

            IgniteFunction<Integer, IgniteNetwork> fact = getNNFactory(histDepthLog, forwardDepth, subpops);

            double lr = 0.5;

            System.out.println("Learning rate is " + lr);

            List<IgniteEvolutionaryOperator> evoOps = Arrays.asList(
                new NodeCrossover(0.2, "nc"),
                new WeightMutation(0.2, lr, "wm"),
                new MutateNodes(0.1, 0.2, lr, "mn")
            );

            int datasetSize = Math.min(maxSamples, (rawData.size() - histDepth - 1) / stepSize);

            int baseIter = 30;

            System.out.println("DS size " + datasetSize);
            GaTrainerCacheInput input = new GaTrainerCacheInput<>(SamplesCache.CACHE_NAME,
                fact,
                datasetSize,
                60,
                evoOps,
                sp -> baseIter * (int)Math.pow(2, sp % subpops), // tree depth drops with grow of subpopulation number, so we can do twice more local ticks with each level drop.
                (in, ign) -> new AdaptableTrainingScore(in.mlDataSet(0, ign)),
                subpops,
                new AddLeaders(0.2)
                    .andThen(new LearningRateAdjuster(null, subpops))
                    .andThen(new BasicStatsCounter()),
                batchPercentage,
                metaoptimizerData -> {
                    BasicStatsCounter.BasicStats stats = metaoptimizerData.get(0).get2();
                    int tick = stats.tick();
                    double ratio = 1.0;
                    if (tick == 1) {
                        ratio = baseIter / 3.0; // 3.0 -- count of initial iterations
                    }
                    long msETA = (long)(stats.currentGlobalTickDuration() * ratio * (maxTicks - (tick - 1)));
                    System.out.println("Current global iteration took " + stats.currentGlobalTickDuration() + "ms, ETA to end is " + (msETA / 1000 / 60) + "mins, " + (msETA / 1000 % 60) + " sec,");
                    return stats.tick() > maxTicks;
                }
            );

            EncogMethodWrapper mdl = new GATrainer<>(ignite).train(input);

            System.out.println("Training took " + (System.currentTimeMillis() - before) + " ms.");

            SamplesCache.getOrCreate(ignite).destroy();

            SequentialRunner runner = new SequentialRunner();

            int size = inputWav.batchs().size();
            int rate = (int)(inputWav.file().getSampleRate() / inputWav.file().getNumChannels());

            runner.add(new MSECalculator());
            runner.add(new PredictedWavWriter(outputWavPath, size, rate, forwardDepth));
            runner.add(new WavTracer((int)(stepSize * csvDown), maxSamples));

            if (outputGenWavPath != null) {
                System.out.println("Added generating in " + outputGenWavPath);
                runner.add(new GeneratedWavWriter(mdl.getM(), outputGenWavPath, size, rate));
            }

            int bestMdlHistDepth = ((BasicNetwork)mdl.getM()).getLayerNeuronCount(0);

            System.out.println("Best model history depth: " + bestMdlHistDepth);
            runner.run(mdl, bestMdlHistDepth, framesInBatch, dataSample);

            PersistorRegistry.getInstance().add(new PersistIgniteNetwork());
            EncogDirectoryPersistence.saveObject(new File(modelsPath + "model" + System.currentTimeMillis() + ".nn"), mdl.getM());
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static int getIntOrDefault(String optionName, int def, CommandLine line) {
        return line.hasOption(optionName) ? Integer.parseInt(line.getOptionValue(optionName)) : def;
    }

    private static double getDoubleOrDefault(String optionName, double def, CommandLine line) {
        return line.hasOption(optionName) ? Double.parseDouble(line.getOptionValue(optionName)) : def;
    }

    /**
     * Build cli options.
     */
    @NotNull private static Options buildOptions() {
        Options options = new Options();

        Option histDepthOpt = OptionBuilder.withArgName("depth_log").withLongOpt("depth_log").hasArg()
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

        Option subPopsOpt = OptionBuilder.withArgName("subpops").withLongOpt("subpops").hasArg().isRequired(false)
            .withDescription("subpopulations count " + SUBPOPS_DEFAULT).create();

        Option wavOutOpt = OptionBuilder.withArgName("out").withLongOpt("out").hasArg().isRequired(false)
            .withDescription("path to predicted wav, default is " + OUTPUT_PRED_WAV_DEFAULT).create();

        Option wavGenOutOpt = OptionBuilder.withArgName("out_gen").withLongOpt("out_gen").hasArg().isRequired(false)
            .withDescription("path of generated wav").create();

        Option batchPercentageOpt = OptionBuilder.withArgName("ds_part").withLongOpt("ds_part").hasArg().isRequired(false)
            .withDescription("Part of dataset which is taken for each global iteration. default value is " + DEFAULT_BATCH_PERCENTAGE).create();

        Option stepSizeOpt = OptionBuilder.withArgName("step_size").withLongOpt("step_size").hasArg().isRequired(false)
            .withDescription("Step size of samples window " + STEP_SIZE_DEFAULT).create();

        Option csvDownsampleRatio = OptionBuilder.withArgName("cdr").withLongOpt("csv_downsample_ratio").hasArg().isRequired(false)
            .withDescription("CSV downsampling ratio " + CSV_DOWNSAMPLE_RATIO).create();

        Option forwardDepthOpt = OptionBuilder.withArgName("fwd").withLongOpt("fwd").hasArg().isRequired(false)
            .withDescription("Forward depth " + FORWARD_DEPTH).create();

        Option modelsDirPathOpt = OptionBuilder.withArgName("mdl").withLongOpt("mdl").hasArg().isRequired(false)
            .withDescription("Directory for storing models, default is " + DEFAULT_MODEL_PATH).create();


        options.addOption(histDepthOpt);
        options.addOption(framesInBatchOpt);
        options.addOption(trainingSamplesOpt);
        options.addOption(trainingDataSampleOpt);
        options.addOption(igniteConfOpt);
        options.addOption(maxSamplesOpt);
        options.addOption(wavOutOpt);
        options.addOption(maxTicksOpt);
        options.addOption(subPopsOpt);
        options.addOption(wavGenOutOpt);
        options.addOption(batchPercentageOpt);
        options.addOption(stepSizeOpt);
        options.addOption(csvDownsampleRatio);
        options.addOption(forwardDepthOpt);
        options.addOption(modelsDirPathOpt);


        return options;
    }

    /** */
    @NotNull private static IgniteFunction<Integer, IgniteNetwork> getNNFactory(int maxLeavesCountLog, int forwardDepth, int subpopsCount) {
        return subPopulation -> {
            return NeuralNetworkUtils.buildTreeLikeNetComplex(maxLeavesCountLog - (subPopulation % subpopsCount), forwardDepth);
//            int treeDepth = maxLeavesCountLog - (subPopulation % 3);
//            IgniteNetwork res = new IgniteNetwork();
//            for (int i = treeDepth; i >= 0; i--)
//                res.addLayer(new BasicLayer(i == 0 ? null : new ActivationSigmoid(), false, (int)Math.pow(2, i)));
//
//            res.getStructure().finalizeStructure();
//
//            for (int i = 0; i < treeDepth - 1; i++) {
//                for (int n = 0; n < res.getLayerNeuronCount(i); n += 2) {
//                    res.dropOutputsFrom(i, n);
//                    res.dropOutputsFrom(i, n + 1);
//
//                    res.enableConnection(i, n, n / 2, true);
//                    res.enableConnection(i, n + 1, n / 2, true);
//                }
//            }
//
//            res.reset();
//
//            return res;
        };
    }
}
