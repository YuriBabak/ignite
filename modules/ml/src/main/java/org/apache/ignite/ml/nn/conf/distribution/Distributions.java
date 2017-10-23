package org.apache.ignite.ml.nn.conf.distribution;

import org.nd4j.linalg.factory.Nd4j;

public class Distributions {
    private Distributions() {
    }

    public static org.nd4j.linalg.api.rng.distribution.Distribution createDistribution(
            Distribution dist) {
        if (dist == null)
            return null;
        if(dist instanceof NormalDistribution) {
            NormalDistribution nd = (NormalDistribution) dist;
            return Nd4j.getDistributions().createNormal(nd.getMean(), nd.getStd());
        }
        throw new RuntimeException("unknown distribution type: " + dist.getClass());
    }
}
