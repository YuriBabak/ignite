package org.apache.ignite.ml.nn.layers;

import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.math.Matrix;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.apache.ignite.ml.nn.gradient.DefaultGradient;
import org.apache.ignite.ml.nn.gradient.Gradient;
import org.apache.ignite.ml.nn.util.Algorithms;
import org.nd4j.linalg.api.ndarray.INDArray;


public class ActivationLayer extends BaseLayer<org.apache.ignite.ml.nn.conf.layers.ActivationLayer> {
    public ActivationLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public double calcL2() {
        return 0;
    }

    @Override
    public double calcL1() {
        return 0;
    }

    @Override
    public Type type() {
        return Type.FEED_FORWARD;
    }

    @Override
    public IgniteBiTuple<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        Matrix activationDerivative;

        String afn = conf().getLayer().getActivationFunction();
        if ("identity".equals(afn)) {
            activationDerivative = Algorithms.toIgnite(input);
        } else if ("relu".equals(afn)) {
            activationDerivative = Algorithms.applyTo("step", Algorithms.toIgnite(input));
        } else {
            throw new RuntimeException("Unsupported activation function.");
        }

        Matrix d = Algorithms.toIgnite(epsilon);
        d = Algorithms.hadamardProduct(activationDerivative, d);

        return new IgniteBiTuple<>(new DefaultGradient(), Algorithms.toNd4j(d));
    }

    @Override
    public INDArray activate(boolean training) {
        if (input == null) {
            throw new IllegalArgumentException("No null input allowed");
        }

        String afn = conf.getLayer().getActivationFunction();
        if ("identity".equals(afn)) {
            return input;
        } else if ("relu".equals(afn)) {
            Matrix output = Algorithms.toIgnite(input);
            output = Algorithms.applyTo("relu", output);

            return Algorithms.toNd4j(output);
        } else {
            throw new RuntimeException("Unsupported activation function.");
        }
    }

    @Override
    public INDArray params(){
        return null;
    }
}
