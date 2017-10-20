package org.apache.ignite.ml.nn.layers.factory;

import org.apache.ignite.ml.nn.api.ParamInitializer;
import org.apache.ignite.ml.nn.conf.layers.Layer;
import org.apache.ignite.ml.nn.params.ConvolutionParamInitializer;


public class ConvolutionLayerFactory extends DefaultLayerFactory {
    public ConvolutionLayerFactory(Class<? extends Layer> layerConfig) {
        super(layerConfig);
    }

    @Override
    public ParamInitializer initializer() {
        return new ConvolutionParamInitializer();
    }
}
