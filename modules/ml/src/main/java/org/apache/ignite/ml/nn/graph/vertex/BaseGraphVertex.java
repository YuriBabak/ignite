package org.apache.ignite.ml.nn.graph.vertex;

import lombok.Data;
import org.apache.ignite.ml.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;


@Data
public abstract class BaseGraphVertex implements GraphVertex {

    protected ComputationGraph graph;

    protected String vertexName;

    protected int vertexIndex;

    protected VertexIndices[] inputVertices;

    protected VertexIndices[] outputVertices;

    protected INDArray[] inputs;
    protected INDArray[] epsilons;

    protected BaseGraphVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices, VertexIndices[] outputVertices){
        this.graph = graph;
        this.vertexName = name;
        this.vertexIndex = vertexIndex;
        this.inputVertices = inputVertices;
        this.outputVertices = outputVertices;

        this.inputs = new INDArray[(inputVertices != null ? inputVertices.length : 0)];
        this.epsilons = new INDArray[(outputVertices != null ? outputVertices.length : 0)];
    }

    @Override
    public String getVertexName(){
        return vertexName;
    }

    @Override
    public int getVertexIndex(){
        return vertexIndex;
    }

    @Override
    public int getNumInputArrays(){
        return (inputVertices == null ? 0 : inputVertices.length);
    }

    @Override
    public int getNumOutputConnections(){
        return (outputVertices == null ? 0 : outputVertices.length);
    }

    @Override
    public VertexIndices[] getInputVertices(){
        return inputVertices;
    }

    @Override
    public void setInputVertices(VertexIndices[] inputVertices){
        this.inputVertices = inputVertices;
        this.inputs = new INDArray[(inputVertices != null ? inputVertices.length : 0)];
    }

    @Override
    public VertexIndices[] getOutputVertices(){
        return outputVertices;
    }

    @Override
    public void setOutputVertices(VertexIndices[] outputVertices){
        this.outputVertices = outputVertices;
        this.epsilons = new INDArray[(outputVertices != null ? outputVertices.length : 0)];
    }

    @Override
    public boolean isInputVertex(){
        return false;
    }

    @Override
    public void setInput(int inputNumber, INDArray input){
        if(inputNumber >= getNumInputArrays()) {
            throw new IllegalArgumentException("Invalid input number");
        }
        inputs[inputNumber] = input;
    }

    @Override
    public void setError(int errorNumber, INDArray error){
        if(errorNumber >= getNumOutputConnections() ){
            throw new IllegalArgumentException("Invalid error number: " + errorNumber
                    + ", numOutputEdges = " + (outputVertices != null ? outputVertices.length : 0) );
        }
        epsilons[errorNumber] = error;
    }

    @Override
    public boolean canDoForward(){
        for (INDArray input : inputs) {
            if (input == null) {
                return false;
            }
        }
        return true;
    }

    @Override
    public boolean canDoBackward(){
        for (INDArray input : inputs) {
            if (input == null) {
                return false;
            }
        }
        for (INDArray epsilon : epsilons) {
            if (epsilon == null) {
                return false;
            }
        }
        return true;
    }
}
