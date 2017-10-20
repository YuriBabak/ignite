package org.apache.ignite.ml.nn.optimize.api;

import org.nd4j.linalg.api.ndarray.INDArray;


public interface StepFunction {
    void step(INDArray x, INDArray line);
}
