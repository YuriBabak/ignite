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

package org.apache.ignite.ml.encog.evolution.operators;

import java.util.Random;
import org.apache.ignite.ml.encog.NeuralNetworkUtils;
import org.apache.ignite.ml.encog.util.Util;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.train.EvolutionaryAlgorithm;
import org.encog.ml.genetic.MLMethodGenome;
import org.encog.neural.networks.BasicNetwork;

public class MutateNodes extends IgniteEvolutionaryOperator implements HasLearningRate {
    private int nodesToMutateCnt;
    private double learningRate;

    public MutateNodes(int nodesToMutateCnt, double prob, double learningRate, String operatorId) {
        super(prob, operatorId);
        this.learningRate = learningRate;
        this.nodesToMutateCnt = nodesToMutateCnt;
    }

    @Override public void init(EvolutionaryAlgorithm theOwner) {

    }

    @Override public int offspringProduced() {
        return 1;
    }

    @Override public int parentsNeeded() {
        return 1;
    }

    @Override public void performOperation(Random rnd, Genome[] parents, int parentIndex, Genome[] offspring,
        int offspringIdx) {
        BasicNetwork parent = (BasicNetwork)((MLMethodGenome)parents[parentIndex]).getPhenotype();
        BasicNetwork off = (BasicNetwork)parent.clone();
        int[] nodeNumbers = Util.selectKDistinct(NeuralNetworkUtils.innerNeuronsCount(parent), nodesToMutateCnt);

        for (int nodeNumber : nodeNumbers) {
            int[] ln = NeuralNetworkUtils.toXY(parent, nodeNumber, true);
            int layer = ln[0];
            int neuron = ln[1];

            // Mutate inputs
            for (int i = 0; i < off.getLayerNeuronCount(layer - 1); i++) {
                double curWeight = off.getWeight(layer - 1, i, neuron);
                double v = (rnd.nextDouble() - 0.5) * learningRate;
                off.setWeight(layer - 1, i, neuron, curWeight + v);
            }

            // Mutate outputs
            for (int i = 0; i < off.getLayerNeuronCount(layer + 1); i++) {
                double curWeight = off.getWeight(layer, neuron, i);
                double v = (rnd.nextDouble() - 0.5) * learningRate;
                off.setWeight(layer, neuron, i, curWeight + v);
            }
        }

        offspring[offspringIdx] = new MLMethodGenome(off);
    }

    @Override public void setLearningRate(double rate) {
        learningRate = rate;
    }

    @Override public double learningRate() {
        return learningRate;
    }
}
