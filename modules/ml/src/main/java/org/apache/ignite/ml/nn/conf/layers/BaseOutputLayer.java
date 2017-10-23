package org.apache.ignite.ml.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.apache.ignite.ml.nn.util.LossFunction;

@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public abstract class BaseOutputLayer extends FeedForwardLayer {
	protected LossFunction lossFunction;

    protected BaseOutputLayer(Builder builder) {
    	super(builder);
        this.lossFunction = builder.lossFunction;
    }
    
    public static abstract class Builder<T extends Builder<T>> extends FeedForwardLayer.Builder<T> {
        protected LossFunction lossFunction = LossFunction.NEGATIVELOGLIKELIHOOD;

        public Builder() {}
    }
}
