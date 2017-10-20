package org.apache.ignite.ml.nn.conf;

import org.nd4j.linalg.api.ndarray.INDArray;


public interface InputPreProcessor extends Cloneable {
    INDArray preProcess(INDArray input, int miniBatchSize);

    INDArray backprop(INDArray output, int miniBatchSize);

    InputPreProcessor clone();
}
