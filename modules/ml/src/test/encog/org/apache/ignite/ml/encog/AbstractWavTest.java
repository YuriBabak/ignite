package org.apache.ignite.ml.encog;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteDataStreamer;
import org.apache.ignite.configuration.DataPageEvictionMode;
import org.apache.ignite.configuration.IgniteConfiguration;
import org.apache.ignite.configuration.MemoryConfiguration;
import org.apache.ignite.configuration.MemoryPolicyConfiguration;
import org.apache.ignite.ml.Model;
import org.apache.ignite.ml.encog.caches.TestTrainingSetCache;
import org.apache.ignite.ml.encog.wav.WavReader;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.apache.ignite.testframework.junits.IgniteTestResources;
import org.apache.ignite.testframework.junits.common.GridCommonAbstractTest;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.neural.networks.layers.BasicLayer;

public abstract class AbstractWavTest extends GridCommonAbstractTest {
    public static final String MEMORY_POLICY = "Test_Memory_Policy";
    /** Grid instance. */
    protected Ignite ignite;

    protected abstract String getWavLocalPath();
    protected abstract int getNodeCount();

    public AbstractWavTest(){
        super(false);
    }

    /**
     * {@inheritDoc}
     */
    @Override protected void beforeTest() throws Exception {
        ignite = grid(getNodeCount());
    }

    /** {@inheritDoc} */
    @Override protected void beforeTestsStarted() throws Exception {
        for (int i = 1; i <= getNodeCount(); i++)
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

        MemoryPolicyConfiguration mpc = new MemoryPolicyConfiguration();

        mpc.setName(MEMORY_POLICY);
        mpc.setMaxSize(15L * 1024 * 1024 * 1024);
        mpc.setInitialSize(1L * 1024 * 1024 * 1024);
        mpc.setPageEvictionMode(DataPageEvictionMode.RANDOM_2_LRU);

        MemoryConfiguration mc = new MemoryConfiguration();

        mc.setMemoryPolicies(mpc);
        mc.setPageSize(16384);

        configuration.setMemoryConfiguration(mc);

        configuration.setIncludeEventTypes();
        configuration.setPeerClassLoadingEnabled(false);
        configuration.setMetricsUpdateFrequency(2000);

        return configuration;
    }

    /**
     * Load wav into cache.
     *
     * @param wav Wav.
     * @param historyDepth History depth.
     */
    protected void loadIntoCache(List<double[]> wav, int historyDepth, int maxSamples, int lookForwardAt) {
        TestTrainingSetCache.getOrCreate(ignite);

        double msd = 0.0;
        double prev = 0.0;

        try (IgniteDataStreamer<Integer, MLDataPair> stmr = ignite.dataStreamer(TestTrainingSetCache.NAME)) {
            // Stream entries.

            int samplesCnt = wav.size();
            System.out.println("Loading " + samplesCnt + " samples into cache...");
            for (int i = historyDepth; i < samplesCnt - 1 && (i - historyDepth) < maxSamples; i++) {

                // The mean is calculated inefficient
                BasicMLData dataSetEntry = new BasicMLData(wav.subList(i - historyDepth, i).stream().map(doubles ->
                    (Arrays.stream(doubles).sum() / doubles.length + 1) / 2).mapToDouble(d -> d).toArray());

                double[] lable = wav.subList(i + 1, i + 1 + lookForwardAt).stream().map(doubles ->
                    (Arrays.stream(doubles).sum() / doubles.length + 1) / 2).mapToDouble(d -> d).toArray();

//                double[] lable = {(Arrays.stream(rawLable).sum() / rawLable.length + 1) / 2};

                stmr.addData(i - historyDepth, new BasicMLDataPair(dataSetEntry, new BasicMLData(lable)));

                if (i % 10_000 == 0)
                    System.out.println("Lb:" + Arrays.toString(lable));

                if (i > historyDepth)
                    msd += (lable[0] - prev) * (lable[0] - prev);
                prev = lable[0];


                if (i % 5000 == 0)
                    System.out.println("Loaded " + i);
            }
            System.out.println("Done, mean squared delta between sequential data is " + msd / Math.min(samplesCnt - historyDepth, maxSamples));
        }
    }

    protected void calculateError(EncogMethodWrapper model, int rate, int sampleNumber, int historyDepth, int framesInBatch) throws IOException {
        List<double[]> rawData = WavReader.read(getWavLocalPath() + "sample" + sampleNumber + "_rate" + rate + ".wav", framesInBatch).batchs();

        PredictionTracer writer = new PredictionTracer();

        IgniteBiFunction<Model<MLData, double[]>, List<double[]>, Double> errorsPercentage = errorsPercentage(
            sampleNumber, rate, historyDepth, writer);
        Double accuracy = errorsPercentage.apply(model, rawData);
        System.out.println(">>> Errs estimation: " + accuracy);
        System.out.println(">>> Tracing data saved: " + writer);
    }

    protected static IgniteNetwork buildTreeLikeNetComplex(int leavesCountLog, int lookForwardCnt) {
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

    protected IgniteBiFunction<Model<MLData, double[]>, List<double[]>, Double> errorsPercentage(int sample, int rate,
        int historyDepth, BiConsumer<Double, Double> writer){

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

                writer.accept(lable[0], predict[0]);

                cnt += predict[0] * predict[0] - lable[0] * lable[0];
            }

            WavReader.write(getWavLocalPath() + "sample" + sample + "_rate" + rate + "_3.wav", buff, rate / 2);
            WavReader.write(getWavLocalPath() + "sample" + sample + "_rate" + rate + "_gen.wav", genBuff, rate);

            return cnt / samplesCnt;
        };
    }
}
