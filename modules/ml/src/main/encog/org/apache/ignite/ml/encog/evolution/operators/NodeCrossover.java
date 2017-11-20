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
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.genetic.MLMethodGenome;
import org.encog.neural.networks.BasicNetwork;

/**
 * Crossover of two networks with tree-like structure.
 */
public class NodeCrossover extends IgniteEvolutionaryOperator {
    public NodeCrossover(double prob, String operatorId) {
        super(prob, operatorId);
    }

    @Override public int offspringProduced() {
        return 1;
    }

    @Override public int parentsNeeded() {
        return 2;
    }

    @Override public void performOperation(Random rnd, Genome[] parents, int parentIndex, Genome[] offspring,
        int offspringIndex) {

        BasicNetwork parent1 = (BasicNetwork)((MLMethodGenome)parents[0]).getPhenotype();
        BasicNetwork parent2 = (BasicNetwork)((MLMethodGenome)parents[1]).getPhenotype();

        BasicNetwork highest = parent1.getLayerCount() >= parent2.getLayerCount() ? parent1 : parent2;
        BasicNetwork lowest = parent1.getLayerCount() < parent2.getLayerCount() ? parent1 : parent2;

        BasicNetwork child = (BasicNetwork)highest.clone();

        int cnt = highest.getLayerCount();
        int depthDelta = highest.getLayerCount() - lowest.getLayerCount();

        for (int l = 1 + depthDelta; l < cnt - 1; l++) {
            if (l < 0) {
                System.out.println("highest :" + highest.getLayerCount() + "lowest: " + lowest.getLayerCount() + " depthDelta: " + depthDelta);
            }
            for (int n = 0; n < lowest.getLayerNeuronCount(l); n++) {
                // true - first parent, false - second parent
                boolean isFirst = rnd.nextBoolean();

                if (!isFirst) {
                    // Set inputs
                    for (int k = 0; k < highest.getLayerNeuronCount(l - 1); k++)
                        child.setWeight(l - 1, k, n, lowest.getWeight(l - 1 - depthDelta, k, n));

                    // Set outputs
                    for (int k = 0; k < highest.getLayerNeuronCount(l + 1); k++)
                        child.setWeight(l, n, k, lowest.getWeight(l - depthDelta, n, k));
                }
            }
        }

        offspring[offspringIndex] = new MLMethodGenome(child);
    }
}
