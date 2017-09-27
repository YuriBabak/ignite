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

import java.util.List;
import java.util.Random;
import org.apache.ignite.ml.encog.caches.TrainingContext;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.genetic.MLMethodGenome;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.Layer;

/**
 * TODO: add description.
 * TODO: use distribution from ctx.
 */
public class WeightMutation extends IgniteEvolutionaryOperator {
    public WeightMutation(double prob){
        super(prob);
    }

    @Override public int offspringProduced() {
        return 1;
    }

    @Override public int parentsNeeded() {
        return 1;
    }

    @Override public void performOperation(Random rnd, Genome[] parents, int parentIndex, Genome[] offspring,
        int offspringIndex) {
        TrainingContext ctx = context();

        BasicNetwork parent = (BasicNetwork)((MLMethodGenome)parents[parentIndex]).getPhenotype();
        BasicNetwork child = (BasicNetwork)parent.clone();

        List<Layer> layers = parent.getStructure().getLayers();

        double shift = rnd.nextDouble() - 0.5d;

        System.out.println("Shift is "+ shift);

        for (int i = 0; i < layers.size() - 1; i++) {
            Layer parentLayer = layers.get(i);
            for (int j = 0; j < parentLayer.getNeuronCount(); j++) {
                for (int k = 0; k < layers.get(i+1).getNeuronCount(); k++) {
                    double parentWeight = parent.getWeight(i, j, k);

                    child.setWeight(i, j, k, parentWeight + shift);
                }
            }
        }

//        double[] weights = parent.getFlat().getWeights().clone();
//
//        for (int i = 0; i < weights.length; i++)
//            weights[i] += rnd.nextDouble();
//
//        BasicNetwork res = (BasicNetwork)ctx.input().methodFactory().get();
//        res.decodeFromArray(weights);

        offspring[offspringIndex] = new MLMethodGenome(child);
    }
}
