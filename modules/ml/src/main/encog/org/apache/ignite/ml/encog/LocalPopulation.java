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
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import javax.cache.Cache;
import org.apache.ignite.Ignite;
import org.apache.ignite.ml.encog.caches.GenomesCache;
import org.apache.ignite.ml.encog.caches.TrainingContext;
import org.apache.ignite.ml.encog.caches.TrainingContextCache;
import org.encog.ml.MethodFactory;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.population.BasicPopulation;
import org.encog.ml.ea.population.Population;
import org.encog.ml.ea.species.Species;
import org.encog.ml.genetic.MLMethodGenome;
import org.encog.ml.genetic.MLMethodGenomeFactory;

public class LocalPopulation<S, U extends Serializable> {
    private UUID trainingUuid;
    private Ignite ignite;
    // Species number -> species population
    private Map<Integer, Population> population;
    // Species number -> species population
    private Map<Integer, List<GenomesCache.Key>> localKeys = new HashMap<>();

    public LocalPopulation(UUID trainingUuid, Ignite ignite) {
        this.trainingUuid = trainingUuid;
        this.ignite = ignite;
        population = load();
    }

    private Map<Integer, Population> load() {
        Map<Integer, Population> res = new HashMap<>();

        // TODO: temporary we filter genomes for the current training in this simple way. Better to make SQL query by training uuid.
        for (Cache.Entry<GenomesCache.Key, MLMethodGenome> entry : GenomesCache.getOrCreate(ignite).localEntries()) {
            if (entry.getKey().trainingUuid().equals(trainingUuid)) {
                addGenome(entry.getValue(), entry.getKey().subPopulation(), res);

                if (!localKeys.containsKey(entry.getKey().subPopulation()))
                    localKeys.put(entry.getKey().subPopulation(), new LinkedList<>());

                localKeys.get(entry.getKey().subPopulation()).add(entry.getKey());
            }
        }

        res.values().forEach(population -> {
            Species species = population.getSpecies().get(0);
            species.getMembers().sort(Comparator.comparing(Genome::getScore));
            Genome best = species.getMembers().get(0);
            species.setLeader(best);
            population.setBestGenome(best);

            TrainingContext<S, U> ctx = TrainingContextCache.getOrCreate(ignite).get(trainingUuid);
            MethodFactory mlMtdFactory = () -> ctx.input().methodFactory().get();
            population.setGenomeFactory(new MLMethodGenomeFactory(mlMtdFactory, population));
        });

        return res;
    }

    private void addGenome(MLMethodGenome genome, int subPopulation, Map<Integer, Population> m) {
        if (!m.containsKey(subPopulation)) {
            BasicPopulation pop = new BasicPopulation();
            pop.createSpecies();
            m.put(subPopulation, pop);
        }

        Population pop = m.get(subPopulation);
        pop.setPopulationSize(pop.getPopulationSize() + 1);
        pop.getSpecies().get(0).add(genome);
        genome.setPopulation(pop);
    }

    public Map<Integer, Population> get() {
        return population;
    }

    public void rewrite(int subPopulation, List<Genome> toSave) {
        // We do not save least fit genomes
//        assert toSave.size() == localKeys.get(subPopulation).size();
        List<Genome> best = toSave.subList(0, localKeys.get(subPopulation).size());

        Map<GenomesCache.Key, MLMethodGenome> m = new HashMap<>();

        best.forEach(GenomesCache::processForSaving);

        int i = 0;
        for (GenomesCache.Key key : localKeys.get(subPopulation))
            m.put(key, (MLMethodGenome)best.get(i));

        GenomesCache.getOrCreate(ignite).putAll(m);
    }
}