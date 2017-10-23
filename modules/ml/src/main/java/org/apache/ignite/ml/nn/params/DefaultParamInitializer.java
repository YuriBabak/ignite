package org.apache.ignite.ml.nn.params;

import java.util.LinkedHashMap;
import java.util.Map;
import org.apache.ignite.ml.nn.api.ParamInitializer;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.apache.ignite.ml.nn.conf.distribution.Distributions;
import org.apache.ignite.ml.nn.util.Algorithms;
import org.apache.ignite.ml.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.indexing.NDArrayIndex;


public class DefaultParamInitializer implements ParamInitializer {

    public final static String WEIGHT_KEY = "W";
    public final static String BIAS_KEY = "b";

    @Override
    public int numParams(NeuralNetConfiguration conf, boolean backprop) {
        org.apache.ignite.ml.nn.conf.layers.FeedForwardLayer layerConf =
                (org.apache.ignite.ml.nn.conf.layers.FeedForwardLayer) conf.getLayer();
        int nIn = layerConf.getNIn();
        int nOut = layerConf.getNOut();
        return nIn*nOut + nOut;
    }

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParameters) {
        if(!(conf.getLayer() instanceof org.apache.ignite.ml.nn.conf.layers.FeedForwardLayer))
            throw new IllegalArgumentException("unsupported layer type: " + conf.getLayer().getClass().getName());

        int length = numParams(conf,true);
        if(paramsView.length() != length) throw new IllegalStateException("Expected params view of length " + length + ", got length " + paramsView.length());

        org.apache.ignite.ml.nn.conf.layers.FeedForwardLayer layerConf =
                (org.apache.ignite.ml.nn.conf.layers.FeedForwardLayer) conf.getLayer();
        int nIn = layerConf.getNIn();
        int nOut = layerConf.getNOut();

        int nWeightParams = nIn*nOut;
        INDArray weightView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0,nWeightParams));
        INDArray biasView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(nWeightParams, nWeightParams + nOut));


        params.put(WEIGHT_KEY,createWeightMatrix(conf, weightView, initializeParameters));
        params.put(BIAS_KEY,createBias(conf, biasView, initializeParameters));
        conf.addVariable(WEIGHT_KEY);
        conf.addVariable(BIAS_KEY);
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        org.apache.ignite.ml.nn.conf.layers.FeedForwardLayer layerConf =
                (org.apache.ignite.ml.nn.conf.layers.FeedForwardLayer) conf.getLayer();
        int nIn = layerConf.getNIn();
        int nOut = layerConf.getNOut();
        int nWeightParams = nIn*nOut;

        INDArray weightGradientView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0,nWeightParams)).reshape('f',nIn,nOut);
        INDArray biasView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(nWeightParams, nWeightParams + nOut));    //Already a row vector

        Map<String,INDArray> out = new LinkedHashMap<>();
        out.put(WEIGHT_KEY, weightGradientView);
        out.put(BIAS_KEY, biasView);

        return out;
    }


    protected INDArray createBias(NeuralNetConfiguration conf, INDArray biasParamView, boolean initializeParameters) {
        org.apache.ignite.ml.nn.conf.layers.FeedForwardLayer layerConf =
                (org.apache.ignite.ml.nn.conf.layers.FeedForwardLayer) conf.getLayer();
        if(initializeParameters) {
            INDArray ret = Algorithms.valueArrayOf(layerConf.getNOut(), layerConf.getBiasInit());
            biasParamView.assign(ret);
        }
        return biasParamView;
    }


    protected INDArray createWeightMatrix(NeuralNetConfiguration conf, INDArray weightParamView, boolean initializeParameters) {
        org.apache.ignite.ml.nn.conf.layers.FeedForwardLayer layerConf =
                (org.apache.ignite.ml.nn.conf.layers.FeedForwardLayer) conf.getLayer();

        if(initializeParameters) {
            Distribution dist = Distributions.createDistribution(layerConf.getDist());
            INDArray ret = WeightInitUtil.initWeights(
                    layerConf.getNIn(),
                    layerConf.getNOut(),
                    layerConf.getWeightInit(),
                    dist,
                    weightParamView);
            return ret;
        } else {
            return WeightInitUtil.reshapeWeights(new int[]{layerConf.getNIn(), layerConf.getNOut()}, weightParamView);
        }
    }
}
