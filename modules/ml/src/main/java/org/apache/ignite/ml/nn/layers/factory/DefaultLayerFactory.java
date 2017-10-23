package org.apache.ignite.ml.nn.layers.factory;

import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import org.apache.ignite.ml.nn.api.Layer;
import org.apache.ignite.ml.nn.api.LayerFactory;
import org.apache.ignite.ml.nn.api.ParamInitializer;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.apache.ignite.ml.nn.optimize.api.IterationListener;
import org.apache.ignite.ml.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;


public class DefaultLayerFactory implements LayerFactory {
    protected org.apache.ignite.ml.nn.conf.layers.Layer layerConfig;

    public DefaultLayerFactory(Class<? extends org.apache.ignite.ml.nn.conf.layers.Layer> layerConfig) {
        try {
            this.layerConfig = layerConfig.newInstance();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public <E extends Layer> E create(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int index,
                                      INDArray layerParamsView, boolean initializeParams) {
        Layer ret = getInstance(conf);
        ret.setListeners(iterationListeners);
        ret.setParamsViewArray(layerParamsView);
        ret.setParamTable(getParams(conf, layerParamsView, initializeParams));
        ret.setConf(conf);
        return (E) ret;
    }

    protected Layer getInstance(NeuralNetConfiguration conf) {
        if (layerConfig instanceof org.apache.ignite.ml.nn.conf.layers.OutputLayer)
            return new org.apache.ignite.ml.nn.layers.OutputLayer(conf);
        if (layerConfig instanceof org.apache.ignite.ml.nn.conf.layers.ConvolutionLayer)
            return new org.apache.ignite.ml.nn.layers.convolution.ConvolutionLayer(conf);
        if (layerConfig instanceof org.apache.ignite.ml.nn.conf.layers.ActivationLayer)
            return new org.apache.ignite.ml.nn.layers.ActivationLayer(conf);
        throw new RuntimeException("unknown layer type: " + layerConfig);
    }


    protected Map<String, INDArray> getParams(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        ParamInitializer init = initializer();
        Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String, INDArray>());
        init.init(params, conf, paramsView, initializeParams);
        return params;
    }

    @Override
    public ParamInitializer initializer() {
        return new DefaultParamInitializer();
    }
}
