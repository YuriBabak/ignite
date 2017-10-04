package org.apache.ignite.ml.encog;

import org.apache.ignite.ml.Model;

/**
 * Trainer for group training of models.
 */
@FunctionalInterface
public interface GroupTrainer<T, V, I, M extends Model<T, V>> {
    M train(I input);
}
