package org.apache.ignite.ml.nn.conf.distribution;

public class NormalDistribution extends Distribution {

    private double mean, std;

    public NormalDistribution(double mean, double std) {
        this.mean = mean;
        this.std = std;
    }

    public double getMean() {
        return mean;
    }

    public double getStd() {
        return std;
    }
}
