package org.apache.ignite.ml.encog;

import org.apache.ignite.IgniteException;
import org.apache.ignite.compute.ComputeJob;
import org.encog.ml.genetic.MLMethodGenome;

public class UpdatePopulationJob implements ComputeJob {

    private MLMethodGenome lead;

    UpdatePopulationJob(MLMethodGenome lead){
        this.lead = lead;
    }

    @Override public void cancel() {

    }

    //TODO: update new lead for local node
    @Override public Object execute() throws IgniteException {
        return null;
    }
}
