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
import org.apache.ignite.ml.encog.TreeNetwork;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.genetic.MLMethodGenome;
import org.encog.neural.networks.BasicNetwork;
import sun.reflect.generics.tree.Tree;

/**
 * TODO: add description.
 */
public class NodeCrossoverTree extends IgniteEvolutionaryOperator {
    public NodeCrossoverTree(double prob, String operatorId) {
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

        TreeNetwork parent1 = (TreeNetwork)((MLMethodGenome)parents[0]).getPhenotype();
        TreeNetwork parent2 = (TreeNetwork)((MLMethodGenome)parents[1]).getPhenotype();
        TreeNetwork child = (TreeNetwork)parent1.clone();

        int cnt = parent1.getLayerCount();

        for (int l = 1; l < cnt - 1; l++) {
            for (int n = 0; n < parent1.getLayerNeuronCount(l); n++) {
                // true - first parent, false - second parent
                boolean isFirst = rnd.nextBoolean();

                if (!isFirst) {
                    // Set inputs
                    for (int k = 0; k < parent1.getLayerNeuronCount(l - 1); k++)
                        child.setWeight(l - 1, k, n, parent2.getWeight(l - 1, k, n));

                    // Set outputs
                    for (int k = 0; k < parent1.getLayerNeuronCount(l + 1); k++)
                        child.setWeight(l, n, k, parent2.getWeight(l, n, k));
                }
            }
        }

        offspring[offspringIndex] = new MLMethodGenome(child);
    }
}
