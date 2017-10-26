package org.apache.ignite.ml.nn.layers;

import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.nn.api.Layer;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.apache.ignite.ml.nn.gradient.Gradient;
import org.apache.ignite.ml.nn.layers.factory.LayerFactories;
import org.apache.ignite.ml.nn.params.DefaultParamInitializer;
import org.apache.ignite.ml.nn.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.apache.ignite.ml.nn.util.Algorithms;
import org.apache.ignite.ml.math.Matrix;
import org.apache.ignite.ml.math.Vector;

import java.util.*;


public abstract class BaseLayer<LayerConfT extends org.apache.ignite.ml.nn.conf.layers.Layer>
        implements Layer {

    protected INDArray input;
    protected INDArray paramsFlattened;
    protected Map<String,INDArray> params;
    protected Map<String,INDArray> gradientViews;
    protected NeuralNetConfiguration conf;
    protected double score = 0.0;
    protected Gradient gradient;
    protected Collection<IterationListener> iterationListeners = new ArrayList<>();

    public BaseLayer(NeuralNetConfiguration conf) {
        this.conf = conf;
    }

    protected LayerConfT layerConf() {
        return (LayerConfT) this.conf.getLayer();
    }

    @Override
    public void setInput(INDArray input) {
        this.input = input;
    }

    @Override
    public Collection<IterationListener> getListeners() {
        return iterationListeners;
    }

    @Override
    public void setListeners(Collection<IterationListener> listeners) {
        this.iterationListeners = listeners != null ? listeners : new ArrayList<IterationListener>();
    }

    @Override
    public IgniteBiTuple<Gradient,INDArray> backpropGradient(INDArray epsilon) {
        throw new RuntimeException("Method backpropGradient is not implemented.");
    }

    @Override
    public void computeGradientAndScore() {
        throw new RuntimeException("Method computeGradientAndScore is not implemented.");
    }

    @Override
    public INDArray preOutput(INDArray x, TrainingMode training) {
        return preOutput(x,training == TrainingMode.TRAIN);
    }

    @Override
    public double score() {
        return score;
    }

    @Override
    public Gradient gradient() {
        return gradient;
    }

    @Override
    public void setConf(NeuralNetConfiguration conf) {
        this.conf = conf;
    }

    @Override
    public INDArray params() {
        throw new RuntimeException("Method params is not implemented.");
    }

    @Override
    public INDArray getParam(String param) {
        return params.get(param);
    }

    @Override
    public void setParams(INDArray params) {
        if(params == paramsFlattened) return;
        setParams(params,'f');
    }

    protected void setParams(INDArray params, char order) {
        throw new RuntimeException("Method setParams is not implemented.");
    }

    @Override
    public void setParamsViewArray(INDArray params){
        if(this.params != null && params.length() != numParams()) throw new IllegalArgumentException("Invalid input: expect params of length " + numParams()
            + ", got params of length " + params.length());

        this.paramsFlattened = params;
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray gradients) {
        if(this.params != null && gradients.length() != numParams(true)) throw new IllegalArgumentException("Invalid input: expect gradients array of length " + numParams(true)
                + ", got params of length " + gradients.length());

        this.gradientViews = LayerFactories.getFactory(conf).initializer().getGradientsFromFlattened(conf,gradients);
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        this.params = paramTable;
    }

    @Override
    public Map<String, INDArray> paramTable() {
        return params;
    }

    @Override
    public INDArray preOutput(INDArray x, boolean training) {
        if (x == null) {
            throw new IllegalArgumentException("Null input is not allowed.");
        }
        setInput(x);
        return preOutput(training);
    }

    public INDArray preOutput(boolean training) {
        INDArray b = getParam(DefaultParamInitializer.BIAS_KEY);
        INDArray W = getParam(DefaultParamInitializer.WEIGHT_KEY);
        Matrix bT = Algorithms.toIgnite(b);
        Matrix WT = Algorithms.toIgnite(W);

        Matrix retT = Algorithms.toIgnite(input).times(WT);
        // TODO: algorithm below is just row-wise addition with vector.
        for (int i = 0; i != retT.rowSize(); ++i) {
            Vector row = retT.viewRow(i);
            row = row.plus(bT.viewRow(0));
        }
        return Algorithms.toNd4j(retT);
    }

    @Override
    public INDArray activate(boolean training) {
        throw new RuntimeException("Method activate is not implemented.");
    }

    @Override
    public double calcL2() {
    	if(!conf.isUseRegularization() || conf.getLayer().getL2() <= 0.0 ) return 0.0;

        double l2Norm = getParam(DefaultParamInitializer.WEIGHT_KEY).norm2Number().doubleValue();
        return 0.5 * conf.getLayer().getL2() * l2Norm * l2Norm;
    }

    @Override
    public double calcL1() {
    	if(!conf.isUseRegularization() || conf.getLayer().getL1()  <= 0.0 ) return 0.0;
        return conf.getLayer().getL1() * getParam(DefaultParamInitializer.WEIGHT_KEY).norm1Number().doubleValue();
    }

    @Override
    public int batchSize() {
        return input.size(0);
    }

    @Override
    public NeuralNetConfiguration conf() {
        return conf;
    }

    @Override
    public Layer clone() {
        throw new RuntimeException("Method clone is not implemented.");
    }

    @Override
    public Type type() {
        return Type.FEED_FORWARD;
    }

    @Override
    public int numParams() {
        int ret = 0;
        for(INDArray val : params.values())
            ret += val.length();
        return ret;
    }

    @Override
    public int numParams(boolean backwards) {
        if(backwards){
            int ret = 0;
            for(Map.Entry<String,INDArray> entry : params.entrySet()){
                ret += entry.getValue().length();
            }
            return ret;
        }
        else
            return numParams();
    }

    @Override
    public IgniteBiTuple<Gradient, Double> gradientAndScore() {
        return new IgniteBiTuple<>(gradient(),score());
    }

    @Override
    public INDArray input() {
        return input;
    }

    @Override
    public int getInputMiniBatchSize(){
    	return input.size(0);
    }
}
