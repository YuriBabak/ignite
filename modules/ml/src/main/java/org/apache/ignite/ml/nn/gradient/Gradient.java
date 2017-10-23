package org.apache.ignite.ml.nn.gradient;

import java.util.Map;
import org.nd4j.linalg.api.ndarray.INDArray;


public interface Gradient {
    Map<String, INDArray> gradientForVariable();

    INDArray gradient();

    INDArray setGradientFor(String variable, INDArray gradient);

    INDArray setGradientFor(String variable, INDArray gradient, Character flatteningOrder);

    Character flatteningOrderForVariable(String variable);
}
