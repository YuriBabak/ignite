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

import java.util.UUID;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.cache.CacheAtomicityMode;
import org.apache.ignite.cache.CacheMode;
import org.apache.ignite.cache.CacheWriteSynchronizationMode;
import org.apache.ignite.cache.affinity.Affinity;
import org.apache.ignite.cache.affinity.AffinityKeyMapped;
import org.apache.ignite.configuration.CacheConfiguration;
import org.apache.ignite.ml.encog.LocalPopulation;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.genetic.MLMethodGenome;

public class GenomesCache {
    public static final String NAME = "GA_GENOMES_CACHE";

    public static Affinity<Key> affinity(Ignite ignite) {
        return ignite.affinity(NAME);
    }

    public static class Key {
        private UUID trainingUuid;
        private UUID genomeUuid;
        @AffinityKeyMapped
        private Integer subPopulation;

        public Key(UUID trainingUuid, UUID genomeUuid, Integer subPopulation) {
            this.trainingUuid = trainingUuid;
            this.genomeUuid = genomeUuid;
            this.subPopulation = subPopulation;
        }

        public UUID trainingUuid() {
            return trainingUuid;
        }

        public UUID genomeUuid() {
            return genomeUuid;
        }

        public Integer subPopulation() {
            return subPopulation;
        }
    }

    // The key of the cache is uuid of training and uuid of genome
    public static IgniteCache<Key, MLMethodGenome> getOrCreate(Ignite ignite) {
        CacheConfiguration<Key, MLMethodGenome> cfg = new CacheConfiguration<>();

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
    }

    public static void processForSaving(Genome genome) {
        genome.setSpecies(null);
        genome.setPopulation(null);
    }
}
