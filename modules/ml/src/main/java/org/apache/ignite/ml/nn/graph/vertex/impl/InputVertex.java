package org.apache.ignite.ml.nn.graph.vertex.impl;

import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.nn.api.Layer;
import org.apache.ignite.ml.nn.gradient.Gradient;
import org.apache.ignite.ml.nn.graph.ComputationGraph;
import org.apache.ignite.ml.nn.graph.vertex.BaseGraphVertex;
import org.apache.ignite.ml.nn.graph.vertex.VertexIndices;
import org.nd4j.linalg.api.ndarray.INDArray;


public class InputVertex extends BaseGraphVertex {
    public InputVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] outputVertices) {
        super(graph, name, vertexIndex, null, outputVertices);
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
    public boolean isInputVertex(){
        return true;
    }

    @Override
    public Layer getLayer() {
        return null;
    }

    @Override
    public INDArray doForward(boolean training) {
        throw new UnsupportedOperationException("Cannot do forward pass for InputVertex");
    }

    @Override
    public IgniteBiTuple<Gradient, INDArray[]> doBackward() {
        throw new UnsupportedOperationException("Cannot do backward pass for InputVertex");
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if(backpropGradientsViewArray != null) throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }
}
