package org.apache.ignite.ml.nn.graph.vertex.impl;

import lombok.Data;
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.nn.api.Layer;
import org.apache.ignite.ml.nn.conf.InputPreProcessor;
import org.apache.ignite.ml.nn.gradient.Gradient;
import org.apache.ignite.ml.nn.graph.ComputationGraph;
import org.apache.ignite.ml.nn.graph.vertex.BaseGraphVertex;
import org.apache.ignite.ml.nn.graph.vertex.VertexIndices;
import org.apache.ignite.ml.nn.layers.BaseOutputLayer;
import org.nd4j.linalg.api.ndarray.INDArray;


@Data
public class LayerVertex extends BaseGraphVertex {
    private Layer layer;
    private InputPreProcessor layerPreProcessor;

    public LayerVertex(ComputationGraph graph, String name, int vertexIndex, Layer layer, InputPreProcessor layerPreProcessor){
        this(graph, name, vertexIndex, null, null, layer, layerPreProcessor);
    }

    public LayerVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices, VertexIndices[] outputVertices,
                        Layer layer, InputPreProcessor layerPreProcessor){
        super(graph,name,vertexIndex,inputVertices,outputVertices);
        this.graph = graph;
        this.vertexName = name;
        this.vertexIndex = vertexIndex;
        this.inputVertices = inputVertices;
        this.outputVertices = outputVertices;
        this.layer = layer;
        this.layerPreProcessor = layerPreProcessor;

        this.inputs = new INDArray[(inputVertices != null ? inputVertices.length : 0)];
        this.epsilons = new INDArray[(outputVertices != null ? outputVertices.length : 0)];
    }

    @Override
    public boolean hasLayer(){
        return true;
    }

    @Override
    public boolean isOutputVertex(){
        return layer instanceof BaseOutputLayer;
    }

    @Override
    public Layer getLayer(){
        return layer;
    }

    @Override
    public INDArray doForward(boolean training){
        if(!canDoForward()) throw new IllegalStateException("Cannot do forward pass: all inputs not set");

        return layer.activate(training);
    }

    @Override
    public IgniteBiTuple<Gradient,INDArray[]> doBackward(){
        if(!canDoBackward()) throw new IllegalStateException("Cannot do backward pass: all epsilons not set");

        INDArray epsTotal = null;
        if(epsilons != null && epsilons.length == 1 ) epsTotal = epsilons[0];
        else if(epsilons != null && epsilons.length > 1 ){
            epsTotal = epsilons[0].dup();
            for( int i=1; i<epsilons.length; i++ ){
                epsTotal.addi(epsilons[i]);
            }
        }

        IgniteBiTuple<Gradient,INDArray> pair = layer.backpropGradient(epsTotal);    //epsTotal may be null for OutputLayers

        if(layerPreProcessor != null){
            INDArray eps = pair.get2();
            eps = layerPreProcessor.backprop(eps,graph.batchSize());
            pair.set2(eps);
        }

        return new IgniteBiTuple<>(pair.get1(), new INDArray[]{pair.get2()});
    }

    @Override
    public void setInput(int inputNumber, INDArray input){
        if(inputNumber > 0) throw new IllegalArgumentException("Invalid input number: LayerVertex instances have only ");
        inputs[inputNumber] = input;

        INDArray currInput = inputs[0];
        if(layerPreProcessor != null){
            currInput = layerPreProcessor.preProcess(currInput, graph.batchSize());
        }
        layer.setInput(currInput);
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        layer.setBackpropGradientsViewArray(backpropGradientsViewArray);
    }
}
