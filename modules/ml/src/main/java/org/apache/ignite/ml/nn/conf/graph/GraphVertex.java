package org.apache.ignite.ml.nn.conf.graph;

import org.apache.ignite.ml.nn.conf.inputs.InputType;
import org.apache.ignite.ml.nn.conf.inputs.InvalidInputTypeException;
import org.apache.ignite.ml.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;


public abstract class GraphVertex implements Cloneable {
    @Override
    public abstract GraphVertex clone();

    @Override
    public abstract int hashCode();

    public abstract int numParams(boolean backprop);

    public abstract org.apache.ignite.ml.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx,
                                                                               INDArray paramsView, boolean initializeParams);

    public abstract InputType getOutputType(InputType... vertexInputs) throws InvalidInputTypeException;
}
