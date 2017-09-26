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
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import javax.cache.Cache;
import org.apache.ignite.Ignite;
import org.apache.ignite.lang.IgniteBiTuple;
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

public class LocalPopulation {
    private UUID trainingUuid;
    private Ignite ignite;
    private Population population;
    private Set<IgniteBiTuple<UUID, UUID>> localKeys = new HashSet<>();

    public LocalPopulation(UUID trainingUuid, Ignite ignite) {
        this.trainingUuid = trainingUuid;
        this.ignite = ignite;
        population = load();
    }

    private Population load() {
        BasicPopulation res = new BasicPopulation();

        // TODO: temporary we filter genomes for the current training in this simple way. Better to make SQL query by training uuid.
        int genomesCnt = 0;

        Species species = res.createSpecies();

        for (Cache.Entry<IgniteBiTuple<UUID, UUID>, MLMethodGenome> entry : GenomesCache.getOrCreate(ignite).localEntries()) {
            if (entry.getKey().get1().equals(trainingUuid)) {
                species.add(entry.getValue());
                localKeys.add(entry.getKey());
                genomesCnt++;
            }
        }

        species.getMembers().sort(Comparator.comparing(Genome::getScore));
        species.setLeader(species.getMembers().get(0));

        res.setPopulationSize(genomesCnt);

        TrainingContext ctx = TrainingContextCache.getOrCreate(ignite).get(trainingUuid);
        MethodFactory mlMtdFactory = () -> ctx.input().methodFactory().get();
        res.setGenomeFactory(new MLMethodGenomeFactory(mlMtdFactory, res));

        return res;
    }

    public Population get() {
        return population;
    }

    public void rewrite(List<Genome> toSave) {
        assert toSave.size() == localKeys.size();

        Map<IgniteBiTuple<UUID, UUID>, MLMethodGenome> m = new HashMap<>();

        int i = 0;
        for (IgniteBiTuple<UUID, UUID> key : localKeys)
            m.put(key, (MLMethodGenome)toSave.get(i));

        GenomesCache.getOrCreate(ignite).putAll(m);
    }
}
