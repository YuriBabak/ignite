package org.apache.ignite.ml.nn.conf.preprocessor;

import java.util.Arrays;
import lombok.AccessLevel;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import org.apache.ignite.ml.nn.conf.InputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;


@Data
public class FeedForwardToCnnPreProcessor implements InputPreProcessor {
    private int inputHeight;
    private int inputWidth;
    private int numChannels;

    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private int[] shape;


    public FeedForwardToCnnPreProcessor(int inputHeight,
                                        int inputWidth,
                                        int numChannels) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = numChannels;
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize) {
        if(input.ordering() != 'c' || !Shape.strideDescendingCAscendingF(input)) input = input.dup('c');

        this.shape = input.shape();
        if(input.shape().length == 4)
            return input;
        if(input.columns() != inputWidth * inputHeight * numChannels)
            throw new IllegalArgumentException("Invalid input: expect output columns must be equal to rows " + inputHeight
                    + " x columns " + inputWidth + " x channels " + numChannels + " but was instead " + Arrays.toString(input.shape()));

        return input.reshape('c',input.size(0),numChannels,inputHeight,inputWidth);
    }

    @Override
    public INDArray backprop(INDArray epsilons, int miniBatchSize){
        if(epsilons.ordering() != 'c' || !Shape.strideDescendingCAscendingF(epsilons)) epsilons = epsilons.dup('c');

        if(shape == null || ArrayUtil.prod(shape) != epsilons.length()) {
            if(epsilons.rank() == 2) return epsilons;   //should never happen

            return epsilons.reshape('c',epsilons.size(0), numChannels, inputHeight, inputWidth);
        }

        return epsilons.reshape('c',shape);
    }

    @Override
    public FeedForwardToCnnPreProcessor clone() {
        try {
            FeedForwardToCnnPreProcessor clone = (FeedForwardToCnnPreProcessor) super.clone();
            if(clone.shape != null) clone.shape = clone.shape.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
}
