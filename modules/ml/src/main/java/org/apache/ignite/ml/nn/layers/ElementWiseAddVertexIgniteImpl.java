package org.apache.ignite.ml.nn.layers;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.impl.ElementWiseVertex;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;


public class ElementWiseAddVertexIgniteImpl extends ElementWiseVertex {
    public ElementWiseAddVertexIgniteImpl(ComputationGraph graph, String name, int vertexIndex) {
        super(graph, name, vertexIndex, Op.Add);
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean _unused) {
        int nInForwardPass = this.inputs.length;

        INDArray[] out = new INDArray[nInForwardPass];

        for(int i = 0; i < nInForwardPass; ++i) {
            out[i] = this.epsilon.dup();
        }

        return new Pair((Object)null, out);
    }
}
