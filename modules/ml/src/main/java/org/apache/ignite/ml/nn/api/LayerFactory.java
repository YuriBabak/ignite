package org.apache.ignite.ml.nn.api;

import java.util.Collection;

import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.apache.ignite.ml.nn.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;


public interface LayerFactory {
    <E extends Layer> E create(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int index,
                               INDArray layerParamsView, boolean initializeParams);

    ParamInitializer initializer();
}
