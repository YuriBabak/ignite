package org.apache.ignite.ml.nn.optimize.solvers;

import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.nn.api.Model;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.apache.ignite.ml.nn.gradient.Gradient;
import org.apache.ignite.ml.nn.optimize.api.IterationListener;
import org.apache.ignite.ml.nn.optimize.api.StepFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;


public class StochasticGradientDescent extends BaseOptimizer {
    public StochasticGradientDescent(NeuralNetConfiguration conf, StepFunction stepFunction, Collection<IterationListener> iterationListeners, Model model) {
        super(conf, stepFunction, iterationListeners, model);
    }

    @Override
    public boolean optimize() {
        for(int i = 0; i < conf.getNumIterations(); i++) {

            IgniteBiTuple<Gradient,Double> pair = gradientAndScore();
            Gradient gradient = pair.get1();

            INDArray params = model.params();
            stepFunction.step(params,gradient.gradient());
            model.setParams(params);

            for(IterationListener listener : iterationListeners)
                listener.iterationDone(model, i);

            checkTerminalConditions(pair.get1().gradient(), oldScore, score, i);

            iteration++;
        }
        return true;
    }
}
