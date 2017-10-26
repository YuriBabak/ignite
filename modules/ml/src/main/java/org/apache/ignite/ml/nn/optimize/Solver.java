package org.apache.ignite.ml.nn.optimize;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.ignite.ml.nn.api.Model;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.apache.ignite.ml.nn.optimize.api.ConvexOptimizer;
import org.apache.ignite.ml.nn.optimize.api.IterationListener;
import org.apache.ignite.ml.nn.optimize.api.StepFunction;
import org.apache.ignite.ml.nn.optimize.solvers.StochasticGradientDescent;
import org.apache.ignite.ml.nn.optimize.stepfunctions.StepFunctions;


public class Solver {
    private NeuralNetConfiguration conf;
    private Collection<IterationListener> listeners;
    private Model model;
    private ConvexOptimizer optimizer;
    private StepFunction stepFunction;

    public void optimize() {
        if(optimizer == null)
            optimizer = getOptimizer();
        optimizer.optimize();

    }

    public ConvexOptimizer getOptimizer() {
        if(optimizer != null) return optimizer;
        switch(conf.getOptimizationAlgo()) {
            case STOCHASTIC_GRADIENT_DESCENT:
                optimizer = new StochasticGradientDescent(conf,stepFunction,listeners,model);
                break;
            default:
                throw new IllegalStateException("No optimizer found");
        }
        return optimizer;
    }

    public void setListeners(Collection<IterationListener> listeners){
        this.listeners = listeners;
        if(optimizer != null ) optimizer.setListeners(listeners);
    }

    public static class Builder {
        private NeuralNetConfiguration conf;
        private Model model;
        private List<IterationListener> listeners = new ArrayList<>();

        public Builder configure(NeuralNetConfiguration conf) {
            this.conf = conf;
            return this;
        }

        public Builder listeners(Collection<IterationListener> listeners) {
            this.listeners.addAll(listeners);
            return this;
        }
        
        public Builder model(Model model) {
            this.model = model;
            return this;
        }

        public Solver build() {
            Solver solver = new Solver();
            solver.conf = conf;
            solver.stepFunction = StepFunctions.createStepFunction(conf.getStepFunction());
            solver.model = model;
            solver.listeners = listeners;
            return solver;
        }
    }
}
