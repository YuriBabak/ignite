/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.ignite.ml.encog;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import org.encog.neural.networks.BasicNetwork;

/**
 * Extension of {@link BasicNetwork} with emulation of topology changes.
 *
 * For this emulation we use {@link Map} of {@link LockKey} and double.
 * If value is 0 - it`s emulate absence of connection between neurons.
 */
public class IgniteNetwork extends BasicNetwork {
    private Map<LockKey, Double> learningMask = new HashMap<>();

    public IgniteNetwork() {
        super();
    }

    /**
     * @param encoded Encoded.
     * @param mask Mask.
     */
    public void encodeToArray(double[] encoded, Map<LockKey, Double> mask) {
        super.encodeToArray(encoded);

        learningMask = mask;
    }

    /**
     * @param learningMask Learning mask.
     */
    public void setLearningMask(Map<LockKey, Double> learningMask) {
        this.learningMask = learningMask;
    }

    /**
     *
     */
    public Map<LockKey, Double> getLearningMask() {
        return learningMask;
    }

    /** {@inheritDoc} */
    @Override public void setWeight(int fromLayer, int fromNeuron, int toNeuron, double val) {
        // TODO: actually lock weights can be inconsistent, think about it.
        double lockVal = learningMask.getOrDefault(new LockKey(fromLayer, fromNeuron), 1d);

        super.setWeight(fromLayer, fromNeuron, toNeuron, val * lockVal);
    }

    /** {@inheritDoc} */
    @Override public double getWeight(int fromLayer, int fromNeuron, int toNeuron) {
        double lockVal = learningMask.getOrDefault(new LockKey(fromLayer, fromNeuron), 1d);

        return Double.compare(lockVal, 0d) == 0 ? 0d : super.getWeight(fromLayer, fromNeuron, toNeuron);
    }

    public void buildMask(){
        for (int i = 0; i < getLayerCount() - 1; i++) {
            for (int j = 0; j < getLayerNeuronCount(i); j++) {
                for (int k = 0; k < getLayerNeuronCount(j); k++)
                    learningMask.put(new LockKey(i, j), 1d);
            }
        }
    }

    public void dropNeuron(int layer, int neuronNumberInLayer){
        validateLayer(layer);

        learningMask.put(new LockKey(layer, neuronNumberInLayer), 0.0d);

        removeWeights(layer, neuronNumberInLayer);
    }

    public void dropOutputsFrom(int layer, int neuronNumberInLayer) {
        if (layer < getLayerCount() - 1) {
            int nl = layer + 1;
            for (int nn = 0; nn < getLayerNeuronCount(nl); nn++)
                setWeight(layer, neuronNumberInLayer, nn, 0.0);
        }
    }

    private void removeWeights(int layer, int neuronNumberInLayer) {
        // remove inputs
        if (layer > 0) {
            int pl = layer - 1;
            for (int pn = 0; pn < getLayerNeuronCount(pl); pn++)
                setWeight(pl, pn, neuronNumberInLayer, 0.0);
        }

        // remove outputs
        if (layer < getLayerCount() - 1) {
            int nl = layer + 1;
            for (int nn = 0; nn < getLayerNeuronCount(nl); nn++)
                setWeight(layer, neuronNumberInLayer, nn, 0.0);
        }
    }

    public void dropNRandomNeurons(int n, Random r){
        for (int i = 0; i < n; i++) {
            int layer = r.nextInt(getLayerCount() - 2) + 1;

            int neuron = r.nextInt(getLayerNeuronCount(layer));

            dropNeuron(layer, neuron);
        }
    }

    public void dropNRandomNeurons(int n) {
        dropNRandomNeurons(n, new Random());
    }

    public void lockNeuron(int layer, int neuronNumberInLayer, double val){
        validateLayer(layer);

        if (val == 0.0)
            dropNeuron(layer, neuronNumberInLayer);
        else
            learningMask.put(new LockKey(layer, neuronNumberInLayer), val);
    }

    public void randomizeLocks(){
        Random random = new Random();
        learningMask = learningMask.entrySet().stream()
            .collect(Collectors.toMap(Map.Entry::getKey, entry -> entry.setValue(random.nextDouble())));
    }

    @Override public void reset() {
        learningMask.clear();
        super.reset();
    }

    /**
     * We can`t drop input or output neuron
     *
     * @param layer Layer.
     */
    private void validateLayer(int layer){
        assert layer > 0;
        assert layer < getLayerCount() - 1;
    }
}
