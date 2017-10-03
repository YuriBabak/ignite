package org.apache.ignite.ml.nn.layers;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.ActivationLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;


public class ActivationLayerIgniteImpl extends ActivationLayer {
    public ActivationLayerIgniteImpl(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        INDArray delta = layerConf().getActivationFn().backprop(input.dup(), epsilon).getFirst();

        return new Pair<>(new DefaultGradient(), delta);
    }
}
