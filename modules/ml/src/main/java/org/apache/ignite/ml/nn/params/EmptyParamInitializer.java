package org.apache.ignite.ml.nn.params;

import java.util.Collections;
import java.util.Map;
import org.apache.ignite.ml.nn.api.ParamInitializer;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;


public class EmptyParamInitializer implements ParamInitializer {
    @Override
    public int numParams(NeuralNetConfiguration conf, boolean backprop) {
        return 0;
    }

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {}

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        return Collections.emptyMap();
    }
}
