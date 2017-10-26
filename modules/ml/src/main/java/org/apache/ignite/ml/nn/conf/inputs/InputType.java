package org.apache.ignite.ml.nn.conf.inputs;

import lombok.*;


public abstract class InputType {
    public enum Type {FF, CNN}

    public abstract Type getType();

    public static InputType feedForward(int size){
        return new InputTypeFeedForward(size);
    }

    public static InputType convolutional(int height, int width, int depth){
        return new InputTypeConvolutional(height,width,depth);
    }

    @AllArgsConstructor @Getter
    public static class InputTypeFeedForward extends InputType{
        private int size;

        @Override
        public Type getType() {
            return Type.FF;
        }
    }

    @AllArgsConstructor @Data  @EqualsAndHashCode(callSuper=false)
    public static class InputTypeConvolutional extends InputType {
        private int height;
        private int width;
        private int depth;

        @Override
        public Type getType() {
            return Type.CNN;
        }
    }
}
