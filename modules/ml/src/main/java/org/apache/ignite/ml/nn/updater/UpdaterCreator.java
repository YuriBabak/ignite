package org.apache.ignite.ml.nn.updater;

import org.apache.ignite.ml.nn.api.Model;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.apache.ignite.ml.nn.conf.Updater;


public class UpdaterCreator {

    private UpdaterCreator() {
    }

    private static org.apache.ignite.ml.nn.api.Updater getUpdater(NeuralNetConfiguration conf) {
        Updater updater = conf.getLayer().getUpdater();

        switch(updater) {
            case NESTEROVS: return new NesterovsUpdater();
            case SGD: return new SgdUpdater();
        }

        return null;
    }

    public static org.apache.ignite.ml.nn.api.Updater getUpdater(Model layer) {
        return getUpdater(layer.conf());
    }
}
