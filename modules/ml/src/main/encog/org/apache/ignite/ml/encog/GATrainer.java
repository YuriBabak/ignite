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
import org.apache.ignite.Ignition;
import org.apache.ignite.cache.CacheAtomicityMode;
import org.apache.ignite.cache.CacheMode;
import org.apache.ignite.cache.CacheWriteSynchronizationMode;
import org.apache.ignite.compute.ComputeTask;
import org.apache.ignite.configuration.CacheConfiguration;
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.Model;
import org.apache.ignite.ml.encog.caches.GenomesCache;
import org.apache.ignite.ml.encog.caches.TrainingContext;
import org.apache.ignite.ml.encog.caches.TrainingContextCache;
import org.apache.ignite.ml.encog.caches.TrainingSetCache;
import org.apache.ignite.ml.math.distributed.CacheUtils;
import org.encog.Encog;
import org.encog.ml.MLMethod;
import org.encog.ml.MethodFactory;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.genome.GenomeFactory;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.genetic.MLMethodGeneticAlgorithm;
import org.encog.ml.genetic.MLMethodGenome;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.TrainingSetScore;
import org.jetbrains.annotations.NotNull;

/**
 * TODO: add description.
 */
public class GATrainer implements GroupTrainer<MLData, double[], GaTrainerInput, EncogMethodWrapper> {
    public static String CACHE = "encog_nets";

    private Ignite ignite;
    private IgniteCache<IgniteBiTuple<UUID, Integer>, BasicNetwork> cache;

    public GATrainer(Ignite ignite) {
        this.ignite = ignite;
    }

    @Override public EncogMethodWrapper train(GaTrainerInput input) {
        cache = newCache();
        GenomesCache.getOrCreate(ignite);

        UUID trainingUUID = UUID.randomUUID();

        // TODO: initialize genome factory
        TrainingContextCache.getOrCreate(ignite).put(trainingUUID, new TrainingContext(
            new GenomeFactory() {
                @Override public Genome factor() {
                    return null;
                }

                @Override public Genome factor(Genome other) {
                    return null;
                }
            },
            input.methodFactory(),
            input.mlDataSet().size()));

        MLMethodGenome lead = null;

//        execute(new InitTask(), null);

        // Here we seed the first generation and make first iteration of algorithm.
        CacheUtils.bcast(GenomesCache.NAME, () -> GATrainer.initialIteration(trainingUUID));

//        while (!isCompleted()) {

            lead = execute(new GroupTrainerTask(), null);
            execute(new UpdatePopulationTask(), lead);
//        }

        return buildIgniteModel(lead);
    }

    @NotNull private static void initialIteration(UUID trainingUUID) {
        Ignite ignite = Ignition.localIgnite();

        IgniteCache<UUID, TrainingContext> cache = TrainingContextCache.getOrCreate(ignite);
        TrainingContext ctx = cache.get(trainingUUID);
        MLDataSet trainingSet = TrainingSetCache.getMLDataSet(ignite, trainingUUID);
        MethodFactory mtdFactory = ctx.getMlMethodFactory();
        TrainingSetScore score = new TrainingSetScore(trainingSet);

        MLMethodGeneticAlgorithm train = new MLMethodGeneticAlgorithm(mtdFactory, score, 100);

        long before = System.currentTimeMillis();
        System.out.println("Doing Initial iteration");
        train.setThreadCount(1);
        train.iteration();
        System.out.println("Done in " + (System.currentTimeMillis() - before));

        train.finishTraining();

//        Encog.getInstance().shutdown();
    }

    private <T, R> R execute(ComputeTask<T, R> task, T arg) {
        return ignite.compute(ignite.cluster().forCacheNodes(CACHE)).execute(task, arg);
    }

    private EncogMethodWrapper buildIgniteModel(MLMethodGenome lead) {
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
