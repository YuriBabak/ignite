package org.apache.ignite.ml.nn.layers;

import lombok.*;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;


@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class ElementWiseAddVertexIgnite extends ElementWiseVertex {
    public ElementWiseAddVertexIgnite() {
        super(Op.Add);
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx,
                                                                      INDArray paramsView, boolean initializeParams) {
        return new ElementWiseAddVertexIgniteImpl(graph, name, idx);
    }
}
