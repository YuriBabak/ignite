package org.apache.ignite.ml.nn.layers;

import java.util.Collection;
import java.util.Map;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class ConvolutionLayerIgnite extends ConvolutionLayer {
    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        ConvolutionLayerIgniteImpl layer = new ConvolutionLayerIgniteImpl(conf);

        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        layer.setParamTable(paramTable);

        return layer;
    }

    protected ConvolutionLayerIgnite(ConvolutionLayer.BaseConvBuilder<?> builder) {
        super(builder);
    }


    public static class Builder extends ConvolutionLayer.Builder {
        public Builder(int[] kernelSize, int[] stride) {
            super(kernelSize, stride);
        }

        public Builder(int... kernelSize) {
            super(kernelSize);
        }

        @Override
        public ConvolutionLayerIgnite build() {
            return new ConvolutionLayerIgnite(this);
        }
    }
}
