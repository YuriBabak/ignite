package org.apache.ignite.ml.nn.layers;

import java.util.Collection;
import lombok.*;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;


@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class ActivationLayerIgnite extends ActivationLayer {
    protected ActivationLayerIgnite(Builder builder) {
        super(builder);
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        return new ActivationLayerIgniteImpl(conf);
    }


    @NoArgsConstructor
    public static class Builder extends ActivationLayer.Builder {
        public ActivationLayerIgnite build() {
            return new ActivationLayerIgnite(this);
        }
    }
}
