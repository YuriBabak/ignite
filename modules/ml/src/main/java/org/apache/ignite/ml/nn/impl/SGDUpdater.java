package org.apache.ignite.ml.nn.impl;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.config.IUpdater;

@Data
@AllArgsConstructor
public class SGDUpdater implements IUpdater {
    private double learningRate;


    @Data
    @AllArgsConstructor
    public class GradientUpdaterImpl implements GradientUpdater<SGDUpdater> {
        private SGDUpdater config;


        @Override
        public void setStateViewArray(INDArray viewArray, int[] gradientShape, char gradientOrder, boolean initialize) {
        }

        @Override
        public void applyUpdater(INDArray gradient, int iteration) {
            gradient.muli(config.getLearningRate());
        }
    }


    @Override
    public long stateSize(long numParams) {
        return 0L;
    }

    @Override
    public void applySchedules(int iteration, double newLearningRate) {
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        if (viewArray != null) {
            throw new IllegalStateException("SGDUpdater does not support view arrays.");
        }
        return new GradientUpdaterImpl(this);
    }

    @Override
    public SGDUpdater clone() {
        return new SGDUpdater(learningRate);
    }
}
