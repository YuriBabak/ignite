package org.apache.ignite.ml.nn.conf.graph;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.apache.ignite.ml.nn.conf.inputs.InputType;
import org.apache.ignite.ml.nn.conf.inputs.InvalidInputTypeException;
import org.apache.ignite.ml.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;


@Data @EqualsAndHashCode(callSuper=false)
public class ElementWiseVertex extends GraphVertex {

    public ElementWiseVertex(Op op) {
        this.op = op;
    }

    public enum Op {Add, Subtract, Product}

    protected Op op;

    @Override
    public ElementWiseVertex clone() {
        return new ElementWiseVertex(op);
    }

    @Override
    public int numParams(boolean backprop){
        return 0;
    }

    @Override
    public org.apache.ignite.ml.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx,
                                                                      INDArray paramsView, boolean initializeParams) {
        org.apache.ignite.ml.nn.graph.vertex.impl.ElementWiseVertex.Op op;
        switch(this.op){
            case Add:
                op = org.apache.ignite.ml.nn.graph.vertex.impl.ElementWiseVertex.Op.Add;
                break;
            default:
                throw new RuntimeException();
        }
        return new org.apache.ignite.ml.nn.graph.vertex.impl.ElementWiseVertex(graph,name,idx,op);
    }

    @Override
    public InputType getOutputType(InputType... vertexInputs) throws InvalidInputTypeException {
        if(vertexInputs.length == 1) return vertexInputs[0];
        InputType first = vertexInputs[0];
        if(first.getType() != InputType.Type.CNN){
            int size = 0;
            for( int i=1; i<vertexInputs.length; i++ ){
                if(vertexInputs[i].getType() != first.getType()){
                    throw new InvalidInputTypeException("Invalid input: ElementWise vertex cannot process activations of different types:"
                        + " first type = " + first.getType() + ", input type " + (i+1) + " = " + vertexInputs[i].getType());
                }
            }
        } else {
            InputType.InputTypeConvolutional firstConv = (InputType.InputTypeConvolutional)first;
            int fd = firstConv.getDepth();
            int fw = firstConv.getWidth();
            int fh = firstConv.getHeight();

            for( int i=1; i<vertexInputs.length; i++ ){
                if(vertexInputs[i].getType() != InputType.Type.CNN){
                    throw new InvalidInputTypeException("Invalid input: ElementWise vertex cannot process activations of different types:"
                            + " first type = " + InputType.Type.CNN + ", input type " + (i+1) + " = " + vertexInputs[i].getType());
                }

                InputType.InputTypeConvolutional otherConv = (InputType.InputTypeConvolutional) vertexInputs[i];

                int od = otherConv.getDepth();
                int ow = otherConv.getWidth();
                int oh = otherConv.getHeight();

                if(fd != od || fw != ow || fh != oh){
                    throw new InvalidInputTypeException("Invalid input: ElementWise vertex cannot process CNN activations of different sizes:"
                            + "first [depth,width,height] = [" + fd + "," + fw + "," + fh + "], input " + i + " = [" + od + "," + ow + "," + oh + "]");
                }
            }
        }
        return first;
    }
}
