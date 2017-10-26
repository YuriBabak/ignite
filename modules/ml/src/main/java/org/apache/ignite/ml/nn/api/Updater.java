package org.apache.ignite.ml.nn.api;

import org.apache.ignite.ml.nn.gradient.Gradient;


public interface Updater extends Cloneable {
    void update(Layer layer, Gradient gradient, int iteration, int miniBatchSize);

    Updater clone();
}
