package org.apache.ignite.ml.nn.optimize.terminations;

import org.apache.ignite.ml.nn.optimize.api.TerminationCondition;
import org.nd4j.linalg.factory.Nd4j;


public class EpsTermination implements TerminationCondition {
    private double eps = 1e-4;
    private double tolerance = Nd4j.EPS_THRESHOLD;

    public EpsTermination() {
    }

    @Override
    public boolean terminate(double cost,double old, Object[] otherParams) {
        if(cost == 0 && old == 0)
           return false;

        if(otherParams.length >= 2) {
            double eps = (double) otherParams[0];
            double tolerance = (double) otherParams[1];
            return 2.0 * Math.abs(old-cost) <= tolerance*(Math.abs(old) + Math.abs(cost) + eps);
        }
        else
            return 2.0 * Math.abs(old  - cost) <= tolerance * (Math.abs(old) + Math.abs(cost) + eps);
    }
}
