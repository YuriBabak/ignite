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
import org.apache.ignite.Ignite;
import org.apache.ignite.ml.encog.GATrainerInput;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.genetic.MLMethodGenome;
import org.encog.neural.flat.FlatNetwork;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.ContainsFlat;

public class Hillclimb extends IgniteEvolutionaryOperator {
    public Hillclimb(double prob, String operatorId) {
        super(prob, operatorId);
    }

    @Override public int offspringProduced() {
        return 1;
    }

    @Override public int parentsNeeded() {
        return 1;
    }

    @Override public void performOperation(Random rnd, Genome[] parents, int parentIndex, Genome[] offspring,
        int offspringIndex) {
        Ignite ignite = ignite();
        GATrainerInput input = input();

        ContainsFlat parent = (ContainsFlat)((MLMethodGenome)parents[parentIndex]).getPhenotype();
        double[] gradient = new GradientCalculator(parent, input.mlDataSet(ignite)).gradient();
        FlatNetwork off = parent.getFlat().clone();

        double[] weights = off.getWeights();

        for (int i = 0; i < weights.length; i++)
            weights[i] += gradient[i];

        BasicNetwork res = (BasicNetwork)input.methodFactory().get();
        res.decodeFromArray(weights);

        offspring[offspringIndex] = new MLMethodGenome(res);
    }
}
