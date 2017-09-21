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

import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import javax.cache.Cache;
import org.apache.ignite.Ignite;
import org.apache.ignite.ml.encog.caches.GenomesCache;
import org.apache.ignite.ml.encog.caches.TrainingContext;
import org.apache.ignite.ml.encog.caches.TrainingContextCache;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.genome.GenomeFactory;
import org.encog.ml.ea.population.BasicPopulation;
import org.encog.ml.ea.population.Population;
import org.encog.ml.ea.species.Species;
import org.encog.ml.genetic.MLMethodGenome;

/**
 * Wrapper around basic population which loads genomes from the cache.
 */
public class CacheBasedPopulation implements Population {
    private BasicPopulation pop;

    public CacheBasedPopulation(UUID trainingUuid, Ignite ignite) {
        TrainingContext context = TrainingContextCache.getOrCreate(ignite).get(trainingUuid);

        GenomeFactory factory = context.genomeFactory();

        pop = new BasicPopulation();

        for (Cache.Entry<UUID, MLMethodGenome> entry : GenomesCache.getOrCreate(ignite).localEntries()) {
            if (entry.getKey().equals(trainingUuid))
                pop.getSpecies().get(0).add(entry.getValue());
        }

        Genome bestGenome = pop.getSpecies().get(0).getMembers().stream().max(Comparator.comparingDouble(Genome::getScore)).get();

        pop.setBestGenome(bestGenome);
        pop.setGenomeFactory(factory);
        pop.setName("Test");
        pop.setPopulationSize(10);
    }

    @Override public void clear() {
        pop.clear();
    }

    @Override public Species createSpecies() {
        return pop.createSpecies();
    }

    @Override public Species determineBestSpecies() {
        return pop.determineBestSpecies();
    }

    @Override public List<Genome> flatten() {
        return pop.flatten();
    }

    @Override public Genome getBestGenome() {
        return pop.getBestGenome();
    }

    @Override public GenomeFactory getGenomeFactory() {
        return pop.getGenomeFactory();
    }

    @Override public int getMaxIndividualSize() {
        return pop.getMaxIndividualSize();
    }

    @Override public int getPopulationSize() {
        return pop.getPopulationSize();
    }

    @Override public List<Species> getSpecies() {
        return pop.getSpecies();
    }

    @Override public void setBestGenome(Genome genome) {
        pop.setBestGenome(genome);
    }

    @Override public void setGenomeFactory(GenomeFactory factory) {
        pop.setGenomeFactory(factory);
    }

    @Override public void setPopulationSize(int i) {
        pop.setPopulationSize(i);
    }

    @Override public int size() {
        return pop.size();
    }

    @Override public void purgeInvalidGenomes() {
        pop.purgeInvalidGenomes();
    }
}
