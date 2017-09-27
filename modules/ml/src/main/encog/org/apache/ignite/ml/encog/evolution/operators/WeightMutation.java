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
import org.apache.ignite.ml.encog.caches.TrainingContext;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.genetic.MLMethodGenome;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.ContainsFlat;

/**
 * TODO: add description.
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

        ContainsFlat parent = (ContainsFlat)((MLMethodGenome)parents[parentIndex]).getPhenotype();

        double[] weights = parent.getFlat().getWeights().clone();

        for (int i = 0; i < weights.length; i++)
            weights[i] += rnd.nextDouble();

        BasicNetwork res = (BasicNetwork)ctx.input().methodFactory().get();
        res.decodeFromArray(weights);

        offspring[parentIndex] = new MLMethodGenome(res);
    }
}
