package org.apache.ignite.ml.nn.optimize.stepfunctions;

import org.apache.ignite.ml.nn.optimize.api.StepFunction;
import org.nd4j.linalg.api.ndarray.INDArray;


public class NegativeGradientStepFunction implements StepFunction {
    @Override
    public void step(INDArray x, INDArray line) {
        x.subi(line);
    }
}
