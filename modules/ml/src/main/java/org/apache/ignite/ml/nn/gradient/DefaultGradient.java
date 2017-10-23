package org.apache.ignite.ml.nn.gradient;

import java.util.LinkedHashMap;
import java.util.Map;
import org.nd4j.linalg.api.ndarray.INDArray;


public class DefaultGradient implements Gradient {
    private Map<String,INDArray> gradients = new LinkedHashMap<>();
    private Map<String,Character> flatteningOrders;
    private INDArray flattenedGradient;

    public DefaultGradient(){ }

    public DefaultGradient(INDArray flattenedGradient){
        this.flattenedGradient = flattenedGradient;
    }

    @Override
    public Map<String, INDArray> gradientForVariable() {
        return gradients;
    }

    @Override
    public INDArray gradient() {
        return flattenedGradient;
    }

    @Override
    public INDArray setGradientFor(String variable, INDArray newGradient) {
        return gradients.put(variable, newGradient);
    }

    @Override
    public INDArray setGradientFor(String variable, INDArray gradient, Character flatteningOrder) {
        INDArray last = setGradientFor(variable,gradient);

        if(flatteningOrder != null){
            if(flatteningOrders == null) flatteningOrders = new LinkedHashMap<>();
            flatteningOrders.put(variable,flatteningOrder);
        }
        return last;
    }

    @Override
    public Character flatteningOrderForVariable(String variable) {
        if(flatteningOrders == null) return null;
        return flatteningOrders.get(variable);
    }
}
