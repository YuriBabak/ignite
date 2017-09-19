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

import java.util.UUID;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.cache.CacheAtomicityMode;
import org.apache.ignite.cache.CacheMode;
import org.apache.ignite.cache.CacheWriteSynchronizationMode;
import org.apache.ignite.compute.ComputeTask;
import org.apache.ignite.configuration.CacheConfiguration;
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.Model;
import org.encog.neural.networks.BasicNetwork;

/**
 * TODO: add description.
 */
public class GATrainer implements GroupTrainer {
    public static String CACHE = "encog_nets";

    private Ignite ignite;
    private IgniteCache<IgniteBiTuple<UUID, Integer>, BasicNetwork> cache;


    public GATrainer(Ignite ignite) {
        this.ignite = ignite;
    }

    @Override public Model train(Object input) {
        cache = newCache();

        IgniteBiTuple<UUID, Integer> lead = null;

        execute(new InitTask(), null);

        while(!isCompleted()) {
            lead = execute(new GroupTrainerTask(), null);

            execute(new UpdatePopulationTask(), lead);
        }

        return buildIgniteModel(lead);
    }

    private <T, R> R execute(ComputeTask<T, R> task, T arg) {
        return ignite.compute(ignite.cluster().forCacheNodes(CACHE)).execute(task, arg);
    }

    private Model buildIgniteModel(IgniteBiTuple<UUID, Integer> lead) {
        return null; //TODO: impl
    }

    private boolean isCompleted() {
        return true; //TODO: impl
    }

    /** */
    private IgniteCache<IgniteBiTuple<UUID, Integer>, BasicNetwork> newCache() {
        CacheConfiguration<IgniteBiTuple<UUID, Integer>, BasicNetwork> cfg = new CacheConfiguration<>();

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
        cfg.setName(CACHE);

        return ignite.getOrCreateCache(cfg);
    }
}
