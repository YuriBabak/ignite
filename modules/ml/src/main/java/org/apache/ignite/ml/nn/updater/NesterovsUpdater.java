package org.apache.ignite.ml.nn.updater;

import org.apache.ignite.ml.nn.api.Layer;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.Nesterovs;


public class NesterovsUpdater extends BaseUpdater {
    @Override
    public GradientUpdater init(String variable, Layer layer) {
        Nesterovs nesterovs = (Nesterovs) updaterForVariable.get(variable);
        if(nesterovs == null) {
            nesterovs = new Nesterovs(layer.conf().getLayer().getMomentum(), layer.conf().getLearningRateByParam(variable));
            updaterForVariable.put(variable,nesterovs);
        }

        return nesterovs;
    }
}
