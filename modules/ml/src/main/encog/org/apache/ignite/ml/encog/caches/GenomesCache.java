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

package org.apache.ignite.ml.encog.caches;

import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import javax.cache.Cache;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.cache.CacheAtomicityMode;
import org.apache.ignite.cache.CacheMode;
import org.apache.ignite.cache.CacheWriteSynchronizationMode;
import org.apache.ignite.configuration.CacheConfiguration;
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.encog.LocalPopulation;
import org.encog.ml.MethodFactory;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.population.BasicPopulation;
import org.encog.ml.ea.population.Population;
import org.encog.ml.ea.species.Species;
import org.encog.ml.genetic.MLMethodGenome;
import org.encog.ml.genetic.MLMethodGenomeFactory;

public class GenomesCache {
    public static final String NAME = "GA_GENOMES_CACHE";

    // The key of the cache is uuid of training and uuid of genome
    public static IgniteCache<IgniteBiTuple<UUID, UUID>, MLMethodGenome> getOrCreate(Ignite ignite) {
        CacheConfiguration<IgniteBiTuple<UUID, UUID>, MLMethodGenome> cfg = new CacheConfiguration<>();

        // Write to primary.
        cfg.setWriteSynchronizationMode(CacheWriteSynchronizationMode.PRIMARY_SYNC);

        // Atomic transactions only.
        cfg.setAtomicityMode(CacheAtomicityMode.ATOMIC);

        // No eviction.
        cfg.setEvictionPolicy(null);

        // No copying of values.
        cfg.setCopyOnRead(false);

        // Cache is partitioned.
        cfg.setCacheMode(CacheMode.PARTITIONED);

        // Random cache name.
        cfg.setName(NAME);

        return ignite.getOrCreateCache(cfg);
    }

    public static LocalPopulation localPopulation(UUID trainingUuid, Ignite ignite) {
        return new LocalPopulation(trainingUuid, ignite);
//        BasicPopulation res = new BasicPopulation();
//
//        // TODO: temporary we filter genomes for the current training in this simple way. Better to make SQL query by training uuid.
//        int genomesCnt = 0;
//
//        Species species = res.createSpecies();
//
//        for (Cache.Entry<IgniteBiTuple<UUID, UUID>, MLMethodGenome> entry : GenomesCache.getOrCreate(ignite).localEntries()) {
//            if (entry.getKey().get1().equals(trainingUuid)) {
//                species.add(entry.getValue());
//                genomesCnt++;
//            }
//        }
//
//        species.getMembers().sort(Comparator.comparing(Genome::getScore));
//        species.setLeader(species.getMembers().get(0));
//
//        res.setPopulationSize(genomesCnt);
//
//        TrainingContext ctx = TrainingContextCache.getOrCreate(ignite).get(trainingUuid);
//        MethodFactory mlMtdFactory = () -> ctx.input().methodFactory().get();
//        res.setGenomeFactory(new MLMethodGenomeFactory(mlMtdFactory, res));
//
//        return res;
    }
}
