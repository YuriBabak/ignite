package org.apache.ignite.ml.nn.graph.vertex.impl;

import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.nn.api.Layer;
import org.apache.ignite.ml.nn.gradient.Gradient;
import org.apache.ignite.ml.nn.graph.ComputationGraph;
import org.apache.ignite.ml.nn.graph.vertex.BaseGraphVertex;
import org.apache.ignite.ml.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.ndarray.INDArray;


public class ElementWiseVertex extends BaseGraphVertex {
    public enum Op {Add}

    private Op op;
    private int nInForwardPass;

    public ElementWiseVertex(ComputationGraph graph, String name, int vertexIndex, Op op){
        this(graph,name,vertexIndex,null,null,op);
    }

    public ElementWiseVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices, VertexIndices[] outputVertices, Op op) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.op = op;
    }

    @Override
    public boolean hasLayer() {
        return false;
    }

    @Override
    public boolean isOutputVertex() {
        return false;
    }

    @Override
    public Layer getLayer() {
        return null;
    }

    @Override
    public INDArray doForward(boolean training) {
        if(!canDoForward()) throw new IllegalStateException("Cannot do forward pass: inputs not set");

        nInForwardPass = inputs.length;
        if(inputs.length == 1) return inputs[0];

        switch(op){
            case Add:
                INDArray sum = inputs[0].dup();
                for( int i=1; i<inputs.length; i++){
                    sum.addi(inputs[i]);
                }
                return sum;
            default:
                throw new UnsupportedOperationException("Unknown op: " + op);
        }
    }

    @Override
    public IgniteBiTuple<Gradient, INDArray[]> doBackward() {
        if(!canDoBackward()) throw new IllegalStateException("Cannot do backward pass: errors not set");

        if(nInForwardPass == 1) return new IgniteBiTuple<>(null,epsilons);

        switch(op){
            case Add:
                INDArray[] out = new INDArray[nInForwardPass];
                out[0] = epsilons[0];
                for( int i=1; i<nInForwardPass; i++ ) out[i] = out[0].dup();
                return new IgniteBiTuple<>(null,out);
            default:
                throw new UnsupportedOperationException("Unknown op: " + op);
        }
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if(backpropGradientsViewArray != null) throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }
}
