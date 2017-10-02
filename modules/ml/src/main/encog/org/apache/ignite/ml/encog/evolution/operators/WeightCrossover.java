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
 * TODO: add description.
 */
public class WeightCrossover extends IgniteEvolutionaryOperator {
    public WeightCrossover(double prob, String operatorId) {
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
        BasicNetwork child = (BasicNetwork)parent1.clone();

        int count = parent1.getLayerCount();

        for (int i = 0; i < count - 1; i++) {
            for (int j = 0; j < parent1.getLayerNeuronCount(i); j++) {
                for (int k = 0; k < parent1.getLayerNeuronCount(i + 1); k++) {
                    // true - first parent, false - second parent
                    boolean isFirst = rnd.nextBoolean();
                    if (!isFirst)
                        child.setWeight(i, j, k, parent2.getWeight(i, j, k));
                }
            }
        }

        offspring[offspringIndex] = new MLMethodGenome(child);
    }
}
