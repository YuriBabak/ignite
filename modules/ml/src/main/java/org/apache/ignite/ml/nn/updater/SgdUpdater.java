package org.apache.ignite.ml.nn.updater;

import org.apache.ignite.ml.nn.api.Layer;
import org.nd4j.linalg.learning.GradientUpdater;


public class SgdUpdater extends BaseUpdater {
    @Override
    public GradientUpdater init(String variable, Layer layer) {
        org.nd4j.linalg.learning.Sgd updater = (org.nd4j.linalg.learning.Sgd) updaterForVariable.get(variable);
        if(updater == null) {
            updater = new org.nd4j.linalg.learning.Sgd(layer.conf().getLearningRateByParam(variable));
            updaterForVariable.put(variable,updater);
        }

        return updater;
    }
}
