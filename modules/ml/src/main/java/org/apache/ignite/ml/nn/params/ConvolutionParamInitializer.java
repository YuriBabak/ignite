package org.apache.ignite.ml.nn.params;


import org.apache.ignite.ml.nn.api.ParamInitializer;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.apache.ignite.ml.nn.conf.distribution.Distributions;
import org.apache.ignite.ml.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.LinkedHashMap;
import java.util.Map;


public class ConvolutionParamInitializer implements ParamInitializer {
    public final static String WEIGHT_KEY = DefaultParamInitializer.WEIGHT_KEY;
    public final static String BIAS_KEY = DefaultParamInitializer.BIAS_KEY;

    @Override
    public int numParams(NeuralNetConfiguration conf, boolean backprop) {
        org.apache.ignite.ml.nn.conf.layers.ConvolutionLayer layerConf =
                (org.apache.ignite.ml.nn.conf.layers.ConvolutionLayer) conf.getLayer();

        int[] kernel = layerConf.getKernelSize();
        int nIn = layerConf.getNIn();
        int nOut = layerConf.getNOut();
        return nIn * nOut * kernel[0] * kernel[1] + nOut;
    }

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        if (((org.apache.ignite.ml.nn.conf.layers.ConvolutionLayer) conf.getLayer()).getKernelSize().length != 2)
            throw new IllegalArgumentException("Filter size must be == 2");

        org.apache.ignite.ml.nn.conf.layers.ConvolutionLayer layerConf =
                (org.apache.ignite.ml.nn.conf.layers.ConvolutionLayer) conf.getLayer();

        int nOut = layerConf.getNOut();

        INDArray biasView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nOut));
        INDArray weightView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(nOut, numParams(conf,true)));

        params.put(BIAS_KEY, createBias(conf, biasView, initializeParams));
        params.put(WEIGHT_KEY, createWeightMatrix(conf, weightView, initializeParams));
        conf.addVariable(WEIGHT_KEY);
        conf.addVariable(BIAS_KEY);
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {

        org.apache.ignite.ml.nn.conf.layers.ConvolutionLayer layerConf =
                (org.apache.ignite.ml.nn.conf.layers.ConvolutionLayer) conf.getLayer();

        int[] kernel = layerConf.getKernelSize();
        int nIn = layerConf.getNIn();
        int nOut = layerConf.getNOut();

        INDArray biasGradientView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nOut));
        INDArray weightGradientView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(nOut, numParams(conf,true)))
                .reshape('c',nOut, nIn, kernel[0], kernel[1]);

        Map<String,INDArray> out = new LinkedHashMap<>();
        out.put(BIAS_KEY, biasGradientView);
        out.put(WEIGHT_KEY, weightGradientView);
        return out;
    }

    protected INDArray createBias(NeuralNetConfiguration conf, INDArray biasView, boolean initializeParams) {
        org.apache.ignite.ml.nn.conf.layers.ConvolutionLayer layerConf =
                (org.apache.ignite.ml.nn.conf.layers.ConvolutionLayer) conf.getLayer();
        if(initializeParams) biasView.assign(layerConf.getBiasInit());
        return biasView;
    }


    protected INDArray createWeightMatrix(NeuralNetConfiguration conf, INDArray weightView, boolean initializeParams) {
        org.apache.ignite.ml.nn.conf.layers.ConvolutionLayer layerConf =
                (org.apache.ignite.ml.nn.conf.layers.ConvolutionLayer) conf.getLayer();
        if(initializeParams) {
            Distribution dist = Distributions.createDistribution(conf.getLayer().getDist());
            int[] kernel = layerConf.getKernelSize();
            return WeightInitUtil.initWeights(new int[]{layerConf.getNOut(), layerConf.getNIn(), kernel[0], kernel[1]},
                    layerConf.getWeightInit(), dist, 'c', weightView);
        } else {
            int[] kernel = layerConf.getKernelSize();
            return WeightInitUtil.reshapeWeights(new int[]{layerConf.getNOut(), layerConf.getNIn(), kernel[0], kernel[1]}, weightView, 'c');
        }
    }
}
