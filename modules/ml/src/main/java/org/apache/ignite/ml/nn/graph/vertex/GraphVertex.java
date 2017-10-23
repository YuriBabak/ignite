package org.apache.ignite.ml.nn.graph.vertex;

import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.nn.api.Layer;
import org.apache.ignite.ml.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;


public interface GraphVertex {
    String getVertexName();

    int getVertexIndex();

    int getNumInputArrays();

    int getNumOutputConnections();

    VertexIndices[] getInputVertices();

    void setInputVertices(VertexIndices[] inputVertices);

    VertexIndices[] getOutputVertices();

    void setOutputVertices(VertexIndices[] outputVertices);

    boolean hasLayer();

    boolean isInputVertex();

    boolean isOutputVertex();

    Layer getLayer();

    void setInput(int inputNumber, INDArray input);

    void setError(int errorNumber, INDArray error);

    boolean canDoForward();

    boolean canDoBackward();

    INDArray doForward(boolean training);

    IgniteBiTuple<Gradient,INDArray[]> doBackward();

    void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray);
}
