package org.apache.ignite.ml.nn.optimize.solvers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.nn.api.Layer;
import org.apache.ignite.ml.nn.api.Model;
import org.apache.ignite.ml.nn.api.Updater;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.apache.ignite.ml.nn.gradient.Gradient;
import org.apache.ignite.ml.nn.graph.ComputationGraph;
import org.apache.ignite.ml.nn.optimize.api.ConvexOptimizer;
import org.apache.ignite.ml.nn.optimize.api.IterationListener;
import org.apache.ignite.ml.nn.optimize.api.StepFunction;
import org.apache.ignite.ml.nn.optimize.api.TerminationCondition;
import org.apache.ignite.ml.nn.optimize.stepfunctions.NegativeGradientStepFunction;
import org.apache.ignite.ml.nn.optimize.terminations.EpsTermination;
import org.apache.ignite.ml.nn.optimize.terminations.ZeroDirection;
import org.apache.ignite.ml.nn.updater.UpdaterCreator;
import org.apache.ignite.ml.nn.updater.graph.ComputationGraphUpdater;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public abstract class BaseOptimizer implements ConvexOptimizer {

    protected NeuralNetConfiguration conf;
    protected int iteration = 0;
    protected static final Logger log = LoggerFactory.getLogger(BaseOptimizer.class);
    protected StepFunction stepFunction;
    protected Collection<IterationListener> iterationListeners = new ArrayList<>();
    protected Collection<TerminationCondition> terminationConditions = new ArrayList<>();
    protected Model model;
    protected Updater updater;
    protected ComputationGraphUpdater computationGraphUpdater;
    protected double score,oldScore;


    public BaseOptimizer(NeuralNetConfiguration conf,StepFunction stepFunction,Collection<IterationListener> iterationListeners,Model model) {
        this(conf, stepFunction, iterationListeners, Arrays.asList(new ZeroDirection(), new EpsTermination()), model);
    }


    public BaseOptimizer(NeuralNetConfiguration conf,StepFunction stepFunction,Collection<IterationListener> iterationListeners,Collection<TerminationCondition> terminationConditions,Model model) {
        this.conf = conf;
        this.stepFunction = (stepFunction != null ? stepFunction : new NegativeGradientStepFunction());
        this.iterationListeners = iterationListeners != null ? iterationListeners : new ArrayList<IterationListener>();
        this.terminationConditions = terminationConditions;
        this.model = model;
    }

    @Override
    public void setListeners(Collection<IterationListener> listeners){
        if(listeners == null) this.iterationListeners = Collections.emptyList();
        else this.iterationListeners = listeners;
    }

    @Override
    public IgniteBiTuple<Gradient,Double> gradientAndScore() {
        oldScore = score;
        model.computeGradientAndScore();
        IgniteBiTuple<Gradient,Double> pair = model.gradientAndScore();
        score = pair.get2();
        updateGradientAccordingToParams(pair.get1(), model, model.batchSize());
        return pair;
    }

    @Override
    public boolean checkTerminalConditions(INDArray gradient, double oldScore, double score, int i){
        for(TerminationCondition condition : terminationConditions){
            if(condition.terminate(score,oldScore,new Object[]{gradient})){
                log.debug("Hit termination condition on iteration {}: score={}, oldScore={}, condition={}", i, score, oldScore, condition);
                return true;
            }
        }
        return false;
    }

    @Override
    public void updateGradientAccordingToParams(Gradient gradient, Model model, int batchSize) {
        if(model instanceof ComputationGraph){
            ComputationGraph graph = (ComputationGraph)model;
            if(computationGraphUpdater == null){
                computationGraphUpdater = new ComputationGraphUpdater(graph);
            }
            computationGraphUpdater.update(graph, gradient, iteration, batchSize);
        } else {

            if (updater == null)
                updater = UpdaterCreator.getUpdater(model);
            Layer layer = (Layer) model;
            updater.update(layer, gradient, iteration, batchSize);
        }
    }
}
