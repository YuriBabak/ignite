package org.apache.ignite.ml.nn.optimize.terminations;

import org.apache.ignite.ml.nn.optimize.api.TerminationCondition;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


public class ZeroDirection implements TerminationCondition {
    @Override
    public boolean terminate(double cost, double oldCost, Object[] otherParams) {
        INDArray gradient = (INDArray) otherParams[0];
        return Nd4j.getBlasWrapper().level1().asum(gradient) == 0.0;
    }
}
