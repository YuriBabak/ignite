package org.apache.ignite.ml.nn.conf.graph;

import java.util.Arrays;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.apache.ignite.ml.nn.conf.InputPreProcessor;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.apache.ignite.ml.nn.conf.inputs.InputType;
import org.apache.ignite.ml.nn.conf.inputs.InvalidInputTypeException;
import org.apache.ignite.ml.nn.conf.layers.ConvolutionLayer;
import org.apache.ignite.ml.nn.conf.layers.FeedForwardLayer;
import org.apache.ignite.ml.nn.conf.layers.Layer;
import org.apache.ignite.ml.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.apache.ignite.ml.nn.graph.ComputationGraph;
import org.apache.ignite.ml.nn.layers.factory.LayerFactories;
import org.nd4j.linalg.api.ndarray.INDArray;


@AllArgsConstructor
@NoArgsConstructor
@Data  @EqualsAndHashCode(callSuper=false)
public class LayerVertex extends GraphVertex {
    private NeuralNetConfiguration layerConf;
    private InputPreProcessor preProcessor;

    @Override
    public GraphVertex clone() {
        return new LayerVertex(layerConf.clone(), (preProcessor != null ? preProcessor.clone() : null));
    }

    @Override
    public int numParams(boolean backprop){
        return LayerFactories.getFactory(layerConf).initializer().numParams(layerConf,backprop);
    }

    @Override
    public org.apache.ignite.ml.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx,
                                                                      INDArray paramsView, boolean initializeParams) {
        return new org.apache.ignite.ml.nn.graph.vertex.impl.LayerVertex(
                graph, name, idx,
                LayerFactories.getFactory(layerConf).create(layerConf, null, idx, paramsView, initializeParams),
                preProcessor);
    }

    @Override
    public InputType getOutputType(InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length != 1) {
            throw new InvalidInputTypeException("LayerVertex expects exactly one input. Got: " + Arrays.toString(vertexInputs));
        }

        Layer layer = layerConf.getLayer();
        if (layer instanceof ConvolutionLayer) {
            InputType.InputTypeConvolutional afterPreProcessor;
            if (preProcessor != null) {
                if (preProcessor instanceof FeedForwardToCnnPreProcessor) {
                    FeedForwardToCnnPreProcessor ffcnn = (FeedForwardToCnnPreProcessor) preProcessor;
                    afterPreProcessor = (InputType.InputTypeConvolutional) InputType.convolutional(ffcnn.getInputHeight(), ffcnn.getInputWidth(), ffcnn.getNumChannels());
                } else {
                    afterPreProcessor = (InputType.InputTypeConvolutional) vertexInputs[0];
                }
            } else {
                afterPreProcessor = (InputType.InputTypeConvolutional) vertexInputs[0];
            }

            int channelsOut;
            int[] kernel;
            int[] stride;
            int[] padding;
            if (layer instanceof ConvolutionLayer) {
                channelsOut = ((ConvolutionLayer) layer).getNOut();
                kernel = ((ConvolutionLayer) layer).getKernelSize();
                stride = ((ConvolutionLayer) layer).getStride();
                padding = ((ConvolutionLayer) layer).getPadding();
            } else {
                throw new RuntimeException("Unsupported layer type.");
            }

            int inHeight = afterPreProcessor.getHeight();
            int inWidth = afterPreProcessor.getWidth();

            int outWidth = (inWidth - kernel[1] + 2 * padding[1]) / stride[1] + 1;
            int outHeight = (inHeight - kernel[0] + 2 * padding[0]) / stride[0] + 1;

            return InputType.convolutional(outHeight,outWidth,channelsOut);
        } else if (layer instanceof FeedForwardLayer) {
            return InputType.feedForward(((FeedForwardLayer) layer).getNOut());
        } else {
            return vertexInputs[0];
        }
    }
}
