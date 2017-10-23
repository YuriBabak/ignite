package org.apache.ignite.ml.nn.api;

import java.util.Collection;
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.nn.gradient.Gradient;
import org.apache.ignite.ml.nn.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;


public interface Layer extends Model {
    enum Type {
        FEED_FORWARD,
        CONVOLUTIONAL,
        SUBSAMPLING,
        NORMALIZATION
    }

    enum TrainingMode {
        TRAIN,
        TEST
    }

    double calcL2();

    double calcL1();

    Type type();

    IgniteBiTuple<Gradient, INDArray> backpropGradient(INDArray epsilon);

    INDArray preOutput(INDArray x,TrainingMode training);

    INDArray preOutput(INDArray x,boolean training);

    INDArray activate(boolean training);

    Layer clone();

    Collection<IterationListener> getListeners();

    void setListeners(Collection<IterationListener> listeners);

    void setInput(INDArray input);

    int getInputMiniBatchSize();
}
