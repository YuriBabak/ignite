package org.apache.ignite.ml.nn.util;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.LogSoftMax;
import org.nd4j.linalg.factory.Nd4j;


public @Data @Builder
class LossCalculation {
    private INDArray labels;
    private INDArray z;

    private double l1,l2;
    private LossFunction lossFunction;
    private boolean useRegularization;
    private boolean miniBatch = false;
    private int miniBatchSize;
    private String activationFn;
    private INDArray preOut;

    public double score(){
        INDArray exampleScores = scoreArray();
        double ret = exampleScores.sumNumber().doubleValue();
        switch(lossFunction){
            case MCXENT:
            case NEGATIVELOGLIKELIHOOD:
                ret *= -1;
                break;
        }

        if (useRegularization) {
            ret += l1 + l2;
        }

        if(miniBatch)
            ret /= (double) miniBatchSize;

        return ret;
    }

    private INDArray scoreArray() {
        INDArray scoreArray;
        switch (lossFunction) {
            case NEGATIVELOGLIKELIHOOD:
            case MCXENT:
                if(preOut != null && "softmax".equals(activationFn)){
                    INDArray logsoftmax = Nd4j.getExecutioner().execAndReturn(new LogSoftMax(preOut.dup()));
                    INDArray sums = labels.mul(logsoftmax);
                    scoreArray = sums;
                } else {
                    throw new RuntimeException("Unsupported activation function.");
                }
                break;
            default:
                throw new RuntimeException("Unsupported loss function.");
        }

        return scoreArray;
    }
}
