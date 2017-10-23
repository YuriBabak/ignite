package org.apache.ignite.ml.nn.api;

import java.util.Map;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;


public interface ParamInitializer {
    int numParams(NeuralNetConfiguration conf, boolean backprop);

    void init(Map<String, INDArray> paramsMap, NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams);

    Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView);
}
