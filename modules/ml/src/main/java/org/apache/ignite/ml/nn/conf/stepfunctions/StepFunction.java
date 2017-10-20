package org.apache.ignite.ml.nn.conf.stepfunctions;


public class StepFunction implements Cloneable {
    @Override
    public StepFunction clone() {
        try {
            StepFunction clone = (StepFunction) super.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
}
