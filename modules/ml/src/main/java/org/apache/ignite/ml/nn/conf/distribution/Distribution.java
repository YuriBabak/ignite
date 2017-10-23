package org.apache.ignite.ml.nn.conf.distribution;


public abstract class Distribution implements Cloneable {
    @Override
    public Distribution clone() {
        try {
            return (Distribution) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
}
