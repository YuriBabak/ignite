package org.apache.ignite.ml.nn.optimize.api;


public interface TerminationCondition {
    boolean terminate(double cost,double oldCost,Object[] otherParams);
}
