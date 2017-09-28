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

import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteException;
import org.apache.ignite.Ignition;
import org.apache.ignite.compute.ComputeJob;
import org.apache.ignite.ml.encog.caches.GenomesCache;
import org.apache.ignite.ml.encog.caches.TrainingContext;
import org.apache.ignite.ml.encog.caches.TrainingContextCache;
import org.apache.ignite.ml.encog.evolution.operators.IgniteEvolutionaryOperator;
import org.encog.Encog;
import org.encog.ml.CalculateScore;
import org.encog.ml.MethodFactory;
import org.encog.ml.ea.population.Population;
import org.encog.ml.genetic.MLMethodGeneticAlgorithm;

public class LocalTrainingTickJob<S, U extends Serializable> implements ComputeJob {
    private UUID trainingUuid;
    private U data;

    public LocalTrainingTickJob(UUID trainingUuid, U data) {
        this.trainingUuid = trainingUuid;
        this.data = data;
    }

    @Override public void cancel() {

    }

    // 1. Load population
    // 2. Do several iterations of evolution
    // 3. Choose best and return it.
    @Override public List<S> execute() throws IgniteException {
        // Load every genome from cache
        Ignite ignite = Ignition.localIgnite();

        LocalPopulation<S, U> locPop = GenomesCache.localPopulation(trainingUuid, ignite);
        Map<Integer, Population> map = locPop.get();
        List<S> res = new LinkedList<>();

        map.entrySet().forEach(entry -> {
            Population population = entry.getValue();
            int subPopulationNum = entry.getKey();

            TrainingContext<S, U> ctx = TrainingContextCache.getOrCreate(ignite).get(trainingUuid);
            MethodFactory mlMtdFactory = () -> ctx.input().methodFactory().get();

            CalculateScore score = ctx.input().scoreCalculator(ctx, ignite);

            MLMethodGeneticAlgorithm training = new MLMethodGeneticAlgorithm(mlMtdFactory, score, population.getSpecies().get(0).getMembers().size());

            training.getGenetic().setPopulation(population);

            List<IgniteEvolutionaryOperator> evoOps = ctx.input().evolutionaryOperators();
            evoOps.forEach(operator -> {
                operator.setIgnite(ignite);
                operator.setContext(ctx);
                training.getGenetic().addOperation(operator.probability(), operator);
            });

            MLMethodGeneticAlgorithm training1 = ctx.input().metaoptimizer().statsHandler(training, data);

            int i = 0;
            while (i < ctx.input().iterationsPerLocalTick()) {
                training1.iteration();
                i++;
            }

            int newSize = training.getGenetic().getPopulation().getSpecies().get(0).getMembers().size();

            System.out.println("New population size: " + newSize);

            training1.finishTraining();

            res.add(ctx.input().metaoptimizer().extractStats(population, ctx));

            int oldSize = GenomesCache.getOrCreate(ignite).size();

            locPop.rewrite(subPopulationNum, training1.getGenetic().getPopulation().getSpecies().get(0).getMembers());

            System.out.println("CS: " + oldSize + "," + GenomesCache.getOrCreate(ignite).size());

            Encog.getInstance().shutdown();
        });

        return res;
    }
}
