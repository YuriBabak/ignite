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

import java.util.Map;
import java.util.Random;
import org.apache.ignite.ml.encog.IgniteNetwork;
import org.apache.ignite.ml.encog.LockKey;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.genetic.MLMethodGenome;

/**
 * Mutate one weight lock(change rate).
 */
public class TopologyMutation extends IgniteEvolutionaryOperator {

    public static final double REDUCE_RATIO = 0.1;

    public TopologyMutation(double prob, String operatorId) {
        super(prob, operatorId);
    }

    /** {@inheritDoc} */
    @Override public int offspringProduced() {
        return 1;
    }

    /** {@inheritDoc} */
    @Override public int parentsNeeded() {
        return 1;
    }

    /** {@inheritDoc} */
    @Override public void performOperation(Random rnd, Genome[] parents, int parentIndex, Genome[] offspring,
        int offspringIndex) {
        IgniteNetwork parent = (IgniteNetwork)((MLMethodGenome)parents[parentIndex]).getPhenotype();
        IgniteNetwork child = (IgniteNetwork)parent.clone();

        Map<LockKey, Double> mask = parent.getLearningMask();

        int lockLayer = rnd.nextInt(parent.getLayerCount() - 1);
        int lockOut = rnd.nextInt(parent.getLayerNeuronCount(lockLayer));
        int lockIn = rnd.nextInt(parent.getLayerNeuronCount(lockLayer + 1));

        double shift = rnd.nextDouble() * REDUCE_RATIO;

        LockKey key = new LockKey(lockLayer, lockOut, lockIn);

        Double maskVal = mask.get(key);

        mask.put(key, maskVal + shift > 0 ? maskVal + shift <= 1 ? maskVal + shift : 1 : 0);

        child.setLearningMask(mask);

        offspring[0] = new MLMethodGenome(child);
    }
}
