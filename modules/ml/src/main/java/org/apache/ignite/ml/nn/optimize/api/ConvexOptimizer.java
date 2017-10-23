package org.apache.ignite.ml.nn.optimize.api;

import java.util.Collection;
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.nn.api.Model;
import org.apache.ignite.ml.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;


public interface ConvexOptimizer {
    void setListeners(Collection<IterationListener> listeners);

    IgniteBiTuple<Gradient,Double> gradientAndScore();

    boolean optimize();

    void updateGradientAccordingToParams(Gradient gradient, Model model, int batchSize);

    boolean checkTerminalConditions(INDArray gradient, double oldScore, double score, int iteration);
}
