package org.apache.ignite.ml.nn.conf.layers;

import lombok.Data;
import lombok.NoArgsConstructor;

import org.apache.ignite.ml.nn.conf.Updater;
import org.apache.ignite.ml.nn.conf.distribution.Distribution;
import org.apache.ignite.ml.nn.weights.WeightInit;


@Data
@NoArgsConstructor
public abstract class Layer implements Cloneable {
    protected String layerName;
    protected String activationFunction;
    protected WeightInit weightInit;
    protected double biasInit;
    protected Distribution dist;
    protected double learningRate;
    protected double momentum;
    protected double l1;
    protected double l2;
    protected Updater updater;


    public Layer(Builder builder) {
        this.layerName = builder.layerName;
    	this.activationFunction = builder.activationFunction;
    	this.weightInit = builder.weightInit;
        this.biasInit = builder.biasInit;
    	this.dist = builder.dist;
        this.learningRate = builder.learningRate;
        this.momentum = builder.momentum;
        this.l1 = builder.l1;
        this.l2 = builder.l2;
        this.updater = builder.updater;
    }

    @Override
    public Layer clone() {
        try {
            Layer clone = (Layer) super.clone();
            if(clone.dist != null) clone.dist = clone.dist.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    public abstract static class Builder<T extends Builder<T>> {
        protected String layerName = null;
        protected String activationFunction = null;
        protected WeightInit weightInit = null;
        protected double biasInit = Double.NaN;
        protected Distribution dist = null;
        protected double learningRate = Double.NaN;
        protected double momentum = Double.NaN;
        protected double l1 = Double.NaN;
        protected double l2 = Double.NaN;
        protected Updater updater = null;


        public T name(String layerName) {
            this.layerName = layerName;
            return (T) this;
        }

        public T activation(String activationFunction) {
            this.activationFunction = activationFunction;
            return (T) this;
        }

        public T l2(double l2){
            this.l2 = l2;
            return (T)this;
        }

        public T updater(Updater updater){
            this.updater = updater;
            return (T) this;
        }

        public abstract <E extends Layer> E build();
    }
}
