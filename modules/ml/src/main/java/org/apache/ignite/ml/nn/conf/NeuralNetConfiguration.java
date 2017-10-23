package org.apache.ignite.ml.nn.conf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.ignite.ml.nn.api.OptimizationAlgorithm;
import org.apache.ignite.ml.nn.conf.distribution.Distribution;
import org.apache.ignite.ml.nn.conf.distribution.NormalDistribution;
import org.apache.ignite.ml.nn.conf.layers.Layer;
import org.apache.ignite.ml.nn.conf.stepfunctions.StepFunction;
import org.apache.ignite.ml.nn.params.DefaultParamInitializer;
import org.apache.ignite.ml.nn.weights.WeightInit;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


@Data
@NoArgsConstructor
public class NeuralNetConfiguration implements Cloneable {

    private static final Logger log = LoggerFactory.getLogger(NeuralNetConfiguration.class);

    protected Layer layer;
    protected boolean miniBatch = true;
    protected int numIterations;
    protected long seed;
    protected OptimizationAlgorithm optimizationAlgo;
    protected List<String> variables = new ArrayList<>();
    protected StepFunction stepFunction;
    protected boolean useRegularization = false;

    protected Map<String,Double> learningRateByParam = new HashMap<>();
    protected Map<String,Double> l1ByParam = new HashMap<>();
    protected Map<String,Double> l2ByParam = new HashMap<>();


    @Override
    public NeuralNetConfiguration clone()  {
        try {
            NeuralNetConfiguration clone = (NeuralNetConfiguration) super.clone();
            if(clone.layer != null) clone.layer = clone.layer.clone();
            if(clone.stepFunction != null) clone.stepFunction = clone.stepFunction.clone();
            if(clone.variables != null ) clone.variables = new ArrayList<>(clone.variables);
            if(clone.learningRateByParam != null ) clone.learningRateByParam = new HashMap<>(clone.learningRateByParam);
            if(clone.l1ByParam != null ) clone.l1ByParam = new HashMap<>(clone.l1ByParam);
            if(clone.l2ByParam != null ) clone.l2ByParam = new HashMap<>(clone.l2ByParam);
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    public List<String> variables() {
        return new ArrayList<>(variables);
    }

    public void addVariable(String variable) {
        if(!variables.contains(variable)) {
            variables.add(variable);
            setLayerParamLR(variable);
        }
    }

    public void setLayerParamLR(String variable){
        double lr = layer.getLearningRate();
        double l1 = variable.substring(0, 1).equals(DefaultParamInitializer.BIAS_KEY) ? 0.0: layer.getL1();
        double l2 = variable.substring(0, 1).equals(DefaultParamInitializer.BIAS_KEY) ? 0.0: layer.getL2();
        learningRateByParam.put(variable, lr);
        l1ByParam.put(variable, l1);
        l2ByParam.put(variable, l2);

    }

    public double getLearningRateByParam(String variable){
        return learningRateByParam.get(variable);
    }

    public double getL1ByParam(String variable ){
        return l1ByParam.get(variable);
    }

    public double getL2ByParam(String variable ){
        return l2ByParam.get(variable);
    }

    public Object[] getExtraArgs() {
        if(layer == null || layer.getActivationFunction() == null) return new Object[0];
        switch( layer.getActivationFunction()) {
            case "relu" :
                return new Object[] { 0 };
            default:
                return new Object [] {};
        }
    }

    @Data
    public static class Builder implements Cloneable {
        protected String activationFunction = "sigmoid";
        protected WeightInit weightInit = WeightInit.XAVIER;
        protected double biasInit = 0.0;
        protected Distribution dist = null;
        protected double learningRate = 1e-1;
        protected double l1 = Double.NaN;
        protected double l2 = Double.NaN;
        protected Updater updater = Updater.SGD;
        protected double momentum = Double.NaN;
        protected Layer layer;
        protected boolean miniBatch = true;
        protected int numIterations = 5;
        protected long seed = System.currentTimeMillis();
        protected boolean useRegularization = false;
        protected OptimizationAlgorithm optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
        protected StepFunction stepFunction = null;

        public Builder layer(Layer layer) {
            this.layer = layer;
            return this;
        }

        public ComputationGraphConfiguration.GraphBuilder graphBuilder(){
            return new ComputationGraphConfiguration.GraphBuilder(this);
        }

        public Builder iterations(int numIterations) {
            this.numIterations = numIterations;
            return this;
        }

        public Builder seed(int seed) {
            this.seed = (long) seed;
            Nd4j.getRandom().setSeed(seed);
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            Nd4j.getRandom().setSeed(seed);
            return this;
        }

        public Builder optimizationAlgo(OptimizationAlgorithm optimizationAlgo) {
            this.optimizationAlgo = optimizationAlgo;
            return this;
        }

        public Builder regularization(boolean useRegularization) {
            this.useRegularization = useRegularization;
            return this;
        }

        @Override
        public Builder clone() {
            try {
                Builder clone = (Builder) super.clone();
                if(clone.layer != null) clone.layer = clone.layer.clone();
                if(clone.stepFunction != null) clone.stepFunction = clone.stepFunction.clone();

                return clone;

            } catch (CloneNotSupportedException e) {
                throw new RuntimeException(e);
            }
        }

        public Builder activation(String activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        public Builder weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return this;
            }

        public Builder dist(Distribution dist) {
            this.dist = dist;
            return this;
        }

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder l1(double l1) {
            this.l1 = l1;
            return this;
        }

        public Builder l2(double l2) {
            this.l2 = l2;
            return this;
        }

        public Builder momentum(double momentum) {
            this.momentum = momentum;
            return this;
        }

        public Builder updater(Updater updater) {
            this.updater = updater;
            return this;
        }

        private void updaterValidation(String layerName){
            switch (layer.getUpdater()) {
                case NESTEROVS:
                    if (Double.isNaN(momentum) && Double.isNaN(layer.getMomentum())) {
                        layer.setMomentum(0.9);
                    }
                    else if (Double.isNaN(layer.getMomentum()))
                        layer.setMomentum(momentum);
                    break;
            }
        }

        private void generalValidation(String layerName){
            if (layer != null) {
                if (useRegularization) {
                    if (!Double.isNaN(l1) && Double.isNaN(layer.getL1()))
                        layer.setL1(l1);
                    if (!Double.isNaN(l2) && Double.isNaN(layer.getL2()))
                        layer.setL2(l2);
                }
                if (Double.isNaN(l2) && Double.isNaN(layer.getL2()))
                    layer.setL2(0.0);
                if (Double.isNaN(l1) && Double.isNaN(layer.getL1()))
                    layer.setL1(0.0);
                if (layer.getWeightInit() == WeightInit.DISTRIBUTION) {
                    if (dist != null && layer.getDist() == null)
                        layer.setDist(dist);
                    else if (dist == null && layer.getDist() == null) {
                        layer.setDist(new NormalDistribution(1e-3, 1));
                    }
                }
            }
        }

        public NeuralNetConfiguration build() {
            NeuralNetConfiguration conf = new NeuralNetConfiguration();
            conf.layer = layer;
            conf.numIterations = numIterations;
            conf.useRegularization = useRegularization;
            conf.optimizationAlgo = optimizationAlgo;
            conf.seed = seed;
            conf.stepFunction = stepFunction;
            conf.miniBatch = miniBatch;
            String layerName;
            if(layer == null || layer.getLayerName() == null ) layerName = "Layer not named";
            else layerName = "Layer " + layer.getLayerName() ;

            if(layer != null ) {
                if (Double.isNaN(layer.getLearningRate())) layer.setLearningRate(learningRate);
                if (Double.isNaN(layer.getL1())) layer.setL1(l1);
                if (Double.isNaN(layer.getL2())) layer.setL2(l2);
                if (layer.getActivationFunction() == null) layer.setActivationFunction(activationFunction);
                if (layer.getWeightInit() == null) layer.setWeightInit(weightInit);
                if (Double.isNaN(layer.getBiasInit())) layer.setBiasInit(biasInit);
                if (layer.getUpdater() == null) layer.setUpdater(updater);
                updaterValidation(layerName);
            }
            generalValidation(layerName);
            return conf;
        }
    }
}
