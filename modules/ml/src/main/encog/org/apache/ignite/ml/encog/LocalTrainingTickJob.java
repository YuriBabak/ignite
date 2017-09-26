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

import java.util.List;
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
import org.encog.ml.MethodFactory;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.ea.population.Population;
import org.encog.ml.genetic.MLMethodGeneticAlgorithm;
import org.encog.ml.genetic.MLMethodGenome;
import org.encog.neural.networks.training.TrainingSetScore;

public class LocalTrainingTickJob implements ComputeJob {
    private UUID trainingUuid;

    public LocalTrainingTickJob(UUID trainingUuid) {
        this.trainingUuid = trainingUuid;
    }

    @Override public void cancel() {

    }

    // 1. Load population
    // 2. Do several iterations of evolution
    // 3. Choose best and return it.
    @Override public MLMethodGenome execute() throws IgniteException {
        // Load every genome from cache
        Ignite ignite = Ignition.localIgnite();

        LocalPopulation locPop = GenomesCache.localPopulation(trainingUuid, ignite);
        Population population = locPop.get();

        TrainingContext ctx = TrainingContextCache.getOrCreate(ignite).get(trainingUuid);
        MethodFactory mlMtdFactory = () -> ctx.input().methodFactory().get();

        MLDataSet trainingSet = ctx.input().mlDataSet(ignite);
        TrainingSetScore score = new TrainingSetScore(trainingSet);

        MLMethodGeneticAlgorithm training = new MLMethodGeneticAlgorithm(mlMtdFactory, score, population.getSpecies().get(0).getMembers().size());

        training.getGenetic().setPopulation(population);

        List<IgniteEvolutionaryOperator> evoOps = ctx.input().evolutionaryOperators();
        evoOps.forEach(operator -> {
            operator.setIgnite(ignite);
            operator.setContext(ctx);
            training.getGenetic().addOperation(operator.probability(), operator);
        });

        int i = 0;
        while (i < ctx.input().iterationsPerLocalTick()) {
            training.iteration();
            i++;
        }

        int newSize = training.getGenetic().getPopulation().getSpecies().get(0).getMembers().size();

        System.out.println("New population size: " + newSize);

        training.finishTraining();

        MLMethodGenome locallyBest = (MLMethodGenome)training.getGenetic().getBestGenome();

        // These fields should not be serialized.
        locallyBest.setPopulation(null);
        locallyBest.setSpecies(null);

        System.out.println("Locally best score is " + locallyBest.getScore());

        int oldSize = GenomesCache.getOrCreate(ignite).size();

        locPop.rewrite(training.getGenetic().getPopulation().getSpecies().get(0).getMembers());

        System.out.println("CS: " + oldSize + "," + GenomesCache.getOrCreate(ignite).size());

        Encog.getInstance().shutdown();

        return locallyBest;
    }
}
