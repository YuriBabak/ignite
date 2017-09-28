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
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;
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
import org.apache.ignite.ml.encog.evolution.operators.IgniteEvolutionaryOperator;
import org.apache.ignite.ml.math.distributed.CacheUtils;
import org.encog.Encog;
import org.encog.ml.MLMethod;
import org.encog.ml.MLRegression;
import org.encog.ml.MethodFactory;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.genetic.MLMethodGeneticAlgorithm;
import org.encog.ml.genetic.MLMethodGenome;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.TrainingSetScore;
import org.jetbrains.annotations.NotNull;

/**
 * TODO: add description.
 */
public class GATrainer<S, U extends Serializable> implements GroupTrainer<MLData, double[], GATrainerInput<? extends MLMethod, S, U>, EncogMethodWrapper> {
    public static String CACHE = "encog_nets";

    private Ignite ignite;
    private IgniteCache<IgniteBiTuple<UUID, Integer>, BasicNetwork> cache;

    private int iteration = 0;
    private MLMethodGenome globalLead = null;

    public GATrainer(Ignite ignite) {
        this.ignite = ignite;
    }

    @Override public EncogMethodWrapper train(GATrainerInput<? extends MLMethod, S, U> input) {
        cache = newCache();
        GenomesCache.getOrCreate(ignite);

        UUID trainingUUID = UUID.randomUUID();

        // TODO: initialize genome factory
        TrainingContextCache.getOrCreate(ignite).put(trainingUUID, new TrainingContext<>(input));

        // Here we seed the first generation and make first iteration of algorithm.
        Collection<List<S>> stats = CacheUtils.bcast(GenomesCache.NAME, () -> GATrainer.initialIteration(trainingUUID));
        U aggregatedStats = input.metaoptimizer().statsAggregator(stats);

        while (!isCompleted())
            aggregatedStats = execute(new GroupTrainerTask<>(input.metaoptimizer()::statsAggregator, aggregatedStats), trainingUUID);

        return buildIgniteModel(globalLead);
    }

    @NotNull private static <S, U extends Serializable> List<S> initialIteration(UUID trainingUUID) {
        Ignite ignite = Ignition.localIgnite();

        IgniteCache<UUID, TrainingContext> cache = TrainingContextCache.getOrCreate(ignite);
        TrainingContext<S, U> ctx = cache.get(trainingUUID);
        MLDataSet trainingSet = ctx.input().mlDataSet(ignite);
        MethodFactory mtdFactory = () -> ctx.input().methodFactory().get();
        TrainingSetScore score = new TrainingSetScore(trainingSet);
        List<IgniteEvolutionaryOperator> evoOps = ctx.input().evolutionaryOperators();

        List<S> res = new LinkedList<>();

        for (int subPopulation = 0; subPopulation < ctx.input().subPopulationsCount(); subPopulation++) {
            // Check if this specNum is mapped to the current node...
            GenomesCache.Key sampleKey = new GenomesCache.Key(trainingUUID, UUID.randomUUID(), subPopulation);
            if (GenomesCache.affinity(ignite).mapKeyToNode(sampleKey) != ignite.cluster().localNode())
                continue;

            MLMethodGeneticAlgorithm train = new MLMethodGeneticAlgorithm(mtdFactory, score, ctx.input().subPopulationSize());

            evoOps.forEach(operator -> {
                operator.setIgnite(ignite);
                operator.setContext(ctx);
                train.getGenetic().addOperation(operator.probability(), operator);
            });

            long before = System.currentTimeMillis();
            System.out.println("Doing Initial iteration for subPopulation " + subPopulation);
            train.setThreadCount(1);
            for (int i = 0; i < 3; i++)
                train.iteration();
            System.out.println("Done in " + (System.currentTimeMillis() - before));

            train.finishTraining();

            res.add(ctx.input().metaoptimizer().extractStats(train.getGenetic().getPopulation(), ctx));

            Encog.getInstance().shutdown();

            // Load all genomes into cache.
            for (Genome genome : train.getGenetic().getPopulation().getSpecies().get(0).getMembers()) {
                MLMethodGenome typedGenome = (MLMethodGenome)genome;

                GenomesCache.processForSaving(genome);
                GenomesCache.getOrCreate(ignite).put(new GenomesCache.Key(trainingUUID, UUID.randomUUID(), subPopulation), typedGenome);
            }
        }

        return res;
    }

    private <T, R> R execute(ComputeTask<T, R> task, T arg) {
        return ignite.compute(ignite.cluster().forCacheNodes(GenomesCache.NAME)).execute(task, arg);
    }

    private EncogMethodWrapper buildIgniteModel(MLMethodGenome lead) {
        return new EncogMethodWrapper((MLRegression)lead.getPhenotype());
    }

    private boolean isCompleted() {
        return iteration++ == 30; //TODO: impl
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
