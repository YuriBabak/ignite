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
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;
import org.apache.ignite.cache.CacheAtomicityMode;
import org.apache.ignite.cache.CacheMode;
import org.apache.ignite.cache.CacheWriteSynchronizationMode;
import org.apache.ignite.compute.ComputeTask;
import org.apache.ignite.configuration.CacheConfiguration;
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.encog.caches.GenomesCache;
import org.apache.ignite.ml.encog.caches.TrainingContext;
import org.apache.ignite.ml.encog.caches.TrainingContextCache;
import org.apache.ignite.ml.math.distributed.CacheUtils;
import org.encog.ml.MLMethod;
import org.encog.ml.MethodFactory;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.population.Population;
import org.encog.ml.genetic.MLMethodGeneticAlgorithm;
import org.encog.ml.genetic.MLMethodGenome;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.TrainingSetScore;
import org.jetbrains.annotations.NotNull;

/**
 * TODO: add description.
 */
public class GATrainer implements GroupTrainer<MLData, double[], GATrainerInput<? extends MLMethod>, EncogMethodWrapper> {
    public static String CACHE = "encog_nets";

    private Ignite ignite;
    private IgniteCache<IgniteBiTuple<UUID, Integer>, BasicNetwork> cache;

    int i = 0;

    public GATrainer(Ignite ignite) {
        this.ignite = ignite;
    }

    @Override public EncogMethodWrapper train(GATrainerInput<? extends MLMethod> input) {
        cache = newCache();
        GenomesCache.getOrCreate(ignite);

        UUID trainingUUID = UUID.randomUUID();

        // TODO: initialize genome factory
        TrainingContextCache.getOrCreate(ignite).put(trainingUUID, new TrainingContext(input));

        // Here we seed the first generation and make first iteration of algorithm.
        CacheUtils.bcast(GenomesCache.NAME, () -> GATrainer.initialIteration(trainingUUID));

        while (!isCompleted()){
            MLMethodGenome lead = execute(new GroupTrainerTask(), trainingUUID);
            System.out.println("Iteration " + i + " complete, globally best score is " + lead.getScore());

            CacheUtils.bcast(GenomesCache.NAME, () -> GATrainer.updatePopulation(trainingUUID, lead));
        }

        return buildIgniteModel(null);
    }

    private static void updatePopulation(UUID trainingUUID, Genome lead) {
        Ignite ignite = Ignition.localIgnite();

        Population population = GenomesCache.localPopulation(trainingUUID, ignite).get();

        List<Genome> newGenomes = TrainingContextCache.getOrCreate(ignite).get(trainingUUID).input().replaceStrategy().getNewGenomes(population, lead);

    }

    @NotNull private static void initialIteration(UUID trainingUUID) {
        Ignite ignite = Ignition.localIgnite();

        IgniteCache<UUID, TrainingContext> cache = TrainingContextCache.getOrCreate(ignite);
        TrainingContext ctx = cache.get(trainingUUID);
        MLDataSet trainingSet = ctx.input().mlDataSet(ignite);
        MethodFactory mtdFactory = () -> ctx.input().methodFactory().get();
        TrainingSetScore score = new TrainingSetScore(trainingSet);

        MLMethodGeneticAlgorithm train = new MLMethodGeneticAlgorithm(mtdFactory, score, 100);

        long before = System.currentTimeMillis();
        System.out.println("Doing Initial iteration");
        train.setThreadCount(1);
        for (int i = 0; i < 3; i++)
            train.iteration();
        System.out.println("Done in " + (System.currentTimeMillis() - before));

        train.finishTraining();

        // Load all genomes into cache
        for (Genome genome : train.getGenetic().getPopulation().getSpecies().get(0).getMembers()) {
            MLMethodGenome typedGenome = (MLMethodGenome)genome;

            // These fields should not be serialized.
            typedGenome.setPopulation(null);
            typedGenome.setSpecies(null);

            GenomesCache.getOrCreate(ignite).put(new IgniteBiTuple<>(trainingUUID, UUID.randomUUID()), typedGenome);
        }

//        Encog.getInstance().shutdown();
    }

    private <T, R> R execute(ComputeTask<T, R> task, T arg) {
        return ignite.compute(ignite.cluster().forCacheNodes(GenomesCache.NAME)).execute(task, arg);
    }

    private EncogMethodWrapper buildIgniteModel(MLMethodGenome lead) {
        return null; //TODO: impl
    }

    private boolean isCompleted() {
        return i++ == 100; //TODO: impl
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
