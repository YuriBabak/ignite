package org.apache.ignite.ml.nn.conf.layers;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;

@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class ActivationLayer extends FeedForwardLayer {
    private ActivationLayer(Builder builder) {
    	super(builder);
    }

    @Override
    public ActivationLayer clone() {
        ActivationLayer clone = (ActivationLayer) super.clone();
        return clone;
    }

    @AllArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder<Builder> {

        @Override
        public ActivationLayer build() {
            return new ActivationLayer(this);
        }
    }
}
