package org.apache.ignite.ml.nn.layers.factory;

import org.apache.ignite.ml.nn.api.LayerFactory;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.apache.ignite.ml.nn.conf.layers.ActivationLayer;
import org.apache.ignite.ml.nn.conf.layers.ConvolutionLayer;
import org.apache.ignite.ml.nn.conf.layers.Layer;


public class LayerFactories {
    private LayerFactories() {}

    public static LayerFactory getFactory(NeuralNetConfiguration conf) {
        return getFactory(conf.getLayer());
    }

    public static LayerFactory getFactory(Layer layer) {
        Class<? extends Layer> clazz = layer.getClass();
        if(ConvolutionLayer.class.isAssignableFrom(clazz))
            return new ConvolutionLayerFactory(clazz);
        else if(ActivationLayer.class.isAssignableFrom(clazz))
            return new EmptyFactory(clazz);

        return new DefaultLayerFactory(clazz);
    }
}
