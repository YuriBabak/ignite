package org.apache.ignite.ml.nn.graph.vertex;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;


@AllArgsConstructor
@EqualsAndHashCode
public class VertexIndices {
    private final int vertexIndex;
    private final int vertexEdgeNumber;


    public int getVertexIndex() {
        return this.vertexIndex;
    }

    public int getVertexEdgeNumber() {
        return this.vertexEdgeNumber;
    }
}
