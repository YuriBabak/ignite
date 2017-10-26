package org.apache.ignite.ml.nn.weights;


import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;


public class WeightInitUtil {
    public static final char DEFAULT_WEIGHT_INIT_ORDER = 'f';

    private WeightInitUtil() {
    }

    public static INDArray initWeights(int[] shape, WeightInit initScheme, Distribution dist, INDArray paramView) {
        return initWeights(shape, initScheme, dist, DEFAULT_WEIGHT_INIT_ORDER, paramView);
    }

    public static INDArray initWeights(int[] shape, WeightInit initScheme, Distribution dist, char order, INDArray paramView) {
        INDArray ret;
        switch (initScheme) {
            case DISTRIBUTION:
                ret = dist.sample(shape);
                break;
            case XAVIER:
                ret = Nd4j.randn(order, shape).divi(FastMath.sqrt(shape[0] + shape[1]));
                break;
            default:
                throw new IllegalStateException("Illegal weight init value: " + initScheme);
        }

        INDArray flat = Nd4j.toFlattened(order, ret);
        if (flat.length() != paramView.length())
            throw new RuntimeException("ParamView length does not match initialized weights length");

        paramView.assign(flat);

        return paramView.reshape(order, shape);
    }

    public static INDArray initWeights(int nIn, int nOut, WeightInit initScheme, Distribution dist, INDArray paramView) {
        return initWeights(new int[]{nIn, nOut}, initScheme, dist, paramView);
    }

    public static INDArray reshapeWeights(int[] shape, INDArray paramsView) {
        return reshapeWeights(shape, paramsView, DEFAULT_WEIGHT_INIT_ORDER);
    }

    public static INDArray reshapeWeights(int[] shape, INDArray paramsView, char flatteningOrder) {
        return paramsView.reshape(flatteningOrder, shape);
    }
}
