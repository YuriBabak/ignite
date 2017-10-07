package org.apache.ignite.ml.encog;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import org.apache.ignite.ml.Model;
import org.apache.ignite.ml.encog.wav.WavReader;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

public class Estimator {
    /**
     * Calculate mean squared error for the given model.
     *
     * @param model Model.
     * @param histDepth History depth, how many frames we will use for each prediction.
     * @param numOfFramesInBatch Number of frames in batch.
     * @param testWavPath Path to the test wav file.
     */
    public double calculateError(EncogMethodWrapper model, int histDepth, int numOfFramesInBatch, String testWavPath ) throws IOException {
        List<double[]> rawData = WavReader.read(testWavPath, numOfFramesInBatch);

        IgniteBiFunction<Model<MLData, double[]>, List<double[]>, Double> errorsPercentage = errorsPercentage(histDepth);
        double accuracy = errorsPercentage.apply(model, rawData);
        System.out.println(">>> Errs estimation: " + accuracy);

        return accuracy;
    }

    private IgniteBiFunction<Model<MLData, double[]>, List<double[]>, Double> errorsPercentage(int histDepth){
        return (model, ds) -> {
            double cnt = 0L;

            int samplesCnt = ds.size();

            for (int i = histDepth; i < samplesCnt-1; i++){

                BasicMLData dataSetEntry = new BasicMLData(ds.subList(i - histDepth, i).stream().map(doubles ->
                    (Arrays.stream(doubles).sum() / doubles.length + 1) / 2).mapToDouble(d -> d).toArray());

                double[] rawLable = ds.get(i + 1);
                double[] lable = {(Arrays.stream(rawLable).sum() / rawLable.length + 1) / 2};

                double[] predict = model.predict(dataSetEntry);

                cnt += predict[0] * predict[0] - lable[0] * lable[0];
            }

            return cnt / samplesCnt;
        };
    }
}
