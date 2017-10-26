package org.apache.ignite.ml.nn.conf.preprocessor;

import lombok.Data;

import org.apache.ignite.ml.nn.conf.InputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;

import java.util.Arrays;


@Data
public class CnnToFeedForwardPreProcessor implements InputPreProcessor {
    private int inputHeight;
    private int inputWidth;
    private int numChannels;

    public CnnToFeedForwardPreProcessor(int inputHeight,
                                        int inputWidth,
                                        int numChannels) {
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numChannels = numChannels;
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize) {
        if(input.rank() == 2) return input; //Should never happen

        if(input.ordering() != 'c' || !Shape.strideDescendingCAscendingF(input)) input = input.dup('c');

        int[] inShape = input.shape();  //[miniBatch,depthOut,outH,outW]
        int[] outShape = new int[]{inShape[0], inShape[1]*inShape[2]*inShape[3]};

        return input.reshape('c',outShape);
    }

    @Override
    public INDArray backprop(INDArray epsilons, int miniBatchSize){
        if(epsilons.ordering() != 'c' || !Shape.strideDescendingCAscendingF(epsilons)) epsilons = epsilons.dup('c');

        if(epsilons.rank() == 4) return epsilons;   //Should never happen

        if(epsilons.columns() != inputWidth * inputHeight * numChannels )
            throw new IllegalArgumentException("Invalid input: expect output columns must be equal to rows " + inputHeight
                    + " x columns " + inputWidth + " x depth " + numChannels +" but was instead " + Arrays.toString(epsilons.shape()));

        return epsilons.reshape('c', epsilons.size(0), numChannels, inputHeight, inputWidth);
    }

    @Override
    public CnnToFeedForwardPreProcessor clone() {
        try {
            CnnToFeedForwardPreProcessor clone = (CnnToFeedForwardPreProcessor) super.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
}
