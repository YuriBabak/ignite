package org.apache.ignite.ml.nn.updater;

import java.util.HashMap;
import java.util.Map;
import org.apache.ignite.ml.nn.api.Layer;
import org.apache.ignite.ml.nn.api.Updater;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.apache.ignite.ml.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.ops.transforms.Transforms;


public abstract class BaseUpdater implements Updater {
    protected Map<String, GradientUpdater> updaterForVariable = new HashMap<>();

    @Override
    public void update(Layer layer, Gradient gradient, int iteration, int miniBatchSize) {
        String paramName;
        INDArray gradientOrig, gradient2;
        GradientUpdater updater;

        preApply(layer, gradient, iteration);
        for (Map.Entry<String, INDArray> gradientPair : gradient.gradientForVariable().entrySet()) {
            paramName = gradientPair.getKey();
            gradientOrig = gradientPair.getValue();

            updater = init(paramName, layer);
            gradient2 = updater.getGradient(gradientOrig, iteration);
            postApply(layer, gradient2, paramName, miniBatchSize);
            gradient.setGradientFor(paramName, gradient2);
        }
    }


    public void postApply(Layer layer, INDArray gradient, String param, int miniBatchSize) {
        NeuralNetConfiguration conf = layer.conf();
        INDArray params = layer.getParam(param);
        if (conf.isUseRegularization() && conf.getL2ByParam(param) > 0)
            gradient.addi(params.mul(conf.getL2ByParam(param)));    //dC/dw = dC0/dw + lambda/n * w where C0 is pre-l2 cost function
        if (conf.isUseRegularization() && conf.getL1ByParam(param) > 0)
            gradient.addi(Transforms.sign(params).muli(conf.getL1ByParam(param)));
        if (conf.isMiniBatch())
            gradient.divi(miniBatchSize);

    }

    public void preApply(Layer layer, Gradient gradient, int iteration) {}


    public abstract GradientUpdater init(String variable, Layer layer);

    @Override
    public Updater clone(){
        Map<String,GradientUpdater> newMap = new HashMap<>();
        for (Map.Entry<String, GradientUpdater> entry : updaterForVariable.entrySet()) {
            newMap.put(entry.getKey(), entry.getValue().getAggregator(true).getUpdater());
        }

        BaseUpdater updater;
        try{
            updater = this.getClass().getConstructor().newInstance();
        }catch (Exception e){
            throw new RuntimeException(e);
        }
        updater.updaterForVariable = newMap;
        return updater;
    }
}
