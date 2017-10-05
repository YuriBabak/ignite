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

package org.apache.ignite.ml.encog.metaoptimizers;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.ignite.ml.encog.caches.GenomesCache;
import org.encog.ml.ea.genome.BasicGenome;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.population.Population;
import org.encog.ml.genetic.MLMethodGeneticAlgorithm;
import org.encog.ml.genetic.MLMethodGenome;

/**
 * Implementation of {@link Metaoptimizer} for sharing leaders between local populations.
 */
public class AddLeaders implements Metaoptimizer<MLMethodGenome, MLMethodGenome> {
    private double additionRatio;

    public AddLeaders(double additionRatio) {
        this.additionRatio = additionRatio;
    }

    @Override public MLMethodGenome initialData(int subPopulation) {
        return null;
    }

    @Override public MLMethodGenome extractStats(Population population, MLMethodGenome prevLeader) {
        MLMethodGenome locallyBest = (MLMethodGenome)population.getBestGenome();
        GenomesCache.processForSaving(locallyBest);
        System.out.println("Locally best score is " + locallyBest.getScore());
        return locallyBest;
    }

    @Override public Map<Integer, MLMethodGenome> statsAggregator(Map<Integer, MLMethodGenome> stats) {
        MLMethodGenome lead = stats.values().stream().min(Comparator.comparingDouble(BasicGenome::getScore)).get();
        System.out.println("Iteration  complete, globally best score is " + lead.getScore());
        Map<Integer, MLMethodGenome> res = new HashMap<>();

        for (int i = 0; i < stats.size(); i++)
            res.put(i, lead);

        return res;
    }

    @Override
    public MLMethodGeneticAlgorithm statsHandler(MLMethodGeneticAlgorithm train, MLMethodGenome best) {
        Population population = train.getGenetic().getPopulation();
        best.setPopulation(population);
        int size = population.getPopulationSize();
        int cntToAdd = (int)(size * additionRatio);

        List<Genome> members = population.getSpecies().get(0).getMembers();
        for (int i = 0; i < cntToAdd; i++)
            members.add(best);

        members.sort(Comparator.comparingDouble(Genome::getScore));

        population.setPopulationSize(cntToAdd + size);

        return train;
    }
}
