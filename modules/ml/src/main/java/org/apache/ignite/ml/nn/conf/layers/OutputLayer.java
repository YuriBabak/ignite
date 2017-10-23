package org.apache.ignite.ml.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.apache.ignite.ml.nn.util.LossFunction;


@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class OutputLayer extends BaseOutputLayer {
    protected OutputLayer(Builder builder) {
    	super(builder);
    }

    @NoArgsConstructor
    public static class Builder extends BaseOutputLayer.Builder<Builder> {

        public Builder(LossFunction lossFunction) {
            this.lossFunction = lossFunction;
        }
        
        @Override
        @SuppressWarnings("unchecked")
        public OutputLayer build() {
            return new OutputLayer(this);
        }
    }
}

