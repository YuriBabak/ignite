package org.apache.ignite.ml.nn.layers.factory;

import org.apache.ignite.ml.nn.api.ParamInitializer;
import org.apache.ignite.ml.nn.conf.layers.Layer;
import org.apache.ignite.ml.nn.params.EmptyParamInitializer;


public class EmptyFactory extends DefaultLayerFactory {
    public EmptyFactory(Class<? extends Layer> layerClazz) {
        super(layerClazz);
    }


    @Override
    public ParamInitializer initializer() {
        return new EmptyParamInitializer();
    }
}
