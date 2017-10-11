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
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import javax.cache.Cache;
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
import org.apache.ignite.ml.encog.caches.InputCache;
import org.apache.ignite.ml.encog.evolution.operators.IgniteEvolutionaryOperator;
import org.apache.ignite.ml.math.distributed.CacheUtils;
import org.apache.ignite.ml.math.util.MapUtil;
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
 * Implementation of group trainer using genetic algorithm.
 */
public class GATrainer<S, U extends Serializable> implements GroupTrainer<MLData, double[], GATrainerInput<? extends MLMethod, S, U>, EncogMethodWrapper> {
    public static String CACHE = "encog_nets";

    private Ignite ignite;
    private IgniteCache<IgniteBiTuple<UUID, Integer>, BasicNetwork> cache;

    /**
     * @param ignite Ignite.
     */
    public GATrainer(Ignite ignite) {
        this.ignite = ignite;
    }

    @Override public EncogMethodWrapper train(GATrainerInput<? extends MLMethod, S, U> input) {
        CacheUtils.setIgnite(ignite);

        cache = newCache();
        GenomesCache.getOrCreate(ignite);

        UUID trainingUUID = UUID.randomUUID();

        InputCache.getOrCreate(ignite).put(trainingUUID, input);

        // Here we seed the first generation and make first iteration of algorithm.
        Collection<Map<Integer, S>> bcast = CacheUtils.bcast(GenomesCache.NAME, () -> GATrainer.initialIteration(trainingUUID));
        Map<Integer, S> stats = bcast.stream().reduce(new HashMap<>(), (m1, m2) -> MapUtil.mergeMaps(m1, m2, (o1, o2) -> o1, HashMap::new));
        Map<Integer, U> aggregatedStats = input.metaoptimizer().statsAggregator(stats);

        while (!input.shouldStop(aggregatedStats)) {
            GroupTrainerTask<S, U> task = new GroupTrainerTask<>(input.metaoptimizer()::statsAggregator, aggregatedStats);
            aggregatedStats = execute(task, trainingUUID);
        }

        MLMethodGenome genome = CacheUtils.bcast(GenomesCache.NAME, () -> GATrainer.collectBest(trainingUUID)).stream().filter(Objects::nonNull).min(Comparator.comparingDouble(MLMethodGenome::getScore)).get();
        return buildIgniteModel(genome);
    }

    private static MLMethodGenome collectBest(UUID trainingUUID) {
        Ignite ignite = Ignition.localIgnite();

        MLMethodGenome res = null;

        for (Cache.Entry<GenomesCache.Key, MLMethodGenome> entry : GenomesCache.getOrCreate(ignite).localEntries()) {
            if (entry.getKey().trainingUuid().equals(trainingUUID) && (res == null || entry.getValue().getScore() < res.getScore()))
                res = entry.getValue();
        }

        return res;
    }

    @NotNull private static <S, U extends Serializable> Map<Integer, S> initialIteration(UUID trainingUUID) {
        Ignite ignite = Ignition.localIgnite();

        IgniteCache<UUID, GATrainerInput> cache = InputCache.getOrCreate(ignite);
        GATrainerInput<?, S, U> input = cache.get(trainingUUID);
        MLDataSet trainingSet = input.mlDataSet(0, ignite);

        TrainingSetScore score = new TrainingSetScore(trainingSet);
        List<IgniteEvolutionaryOperator> evoOps = input.evolutionaryOperators();

        Map<Integer, S> res = new HashMap<>();

        for (int subPopulation = 0; subPopulation < input.subPopulationsCount(); subPopulation++) {
            int sp = subPopulation;
            MethodFactory mtdFactory = () -> input.methodFactory(sp).get();
            // Check if this specNum is mapped to the current node...
            GenomesCache.Key sampleKey = new GenomesCache.Key(trainingUUID, UUID.randomUUID(), subPopulation);
            if (GenomesCache.affinity(ignite).mapKeyToNode(sampleKey) != ignite.cluster().localNode())
                continue;

            System.out.println("Pop size: " + input.subPopulationSize());
            MLMethodGeneticAlgorithm train = new MLMethodGeneticAlgorithm(mtdFactory, score, input.subPopulationSize());

            evoOps.forEach(operator -> {
                operator.setIgnite(ignite);
                operator.setInput(input);
                train.getGenetic().addOperation(operator.probability(), operator);
            });

            long before = System.currentTimeMillis();
            System.out.println("Doing Initial iteration for subPopulation " + subPopulation);
//            train.setThreadCount(1);
            for (int i = 0; i < 3; i++)
                train.iteration();
            System.out.println("Done in " + (System.currentTimeMillis() - before));

            train.finishTraining();

            res.put(subPopulation, input.metaoptimizer().extractStats(subPopulation, train.getGenetic().getPopulation(), input.metaoptimizer().initialData(subPopulation)));

//            Encog.getInstance().shutdown();

            // Load all genomes into cache.
            for (Genome genome : train.getGenetic().getPopulation().getSpecies().get(0).getMembers()) {
                MLMethodGenome typedGenome = (MLMethodGenome)genome;

                GenomesCache.processForSaving(genome);
                GenomesCache.getOrCreate(ignite).put(new GenomesCache.Key(trainingUUID, UUID.randomUUID(), subPopulation), typedGenome);
            }
        }

        return res;
    }

    /**
     * exec compute task over cache.
     *
     * @param task Task.
     * @param arg Argument.
     */
    private <T, R> R execute(ComputeTask<T, R> task, T arg) {
        return ignite.compute(ignite.cluster().forCacheNodes(GenomesCache.NAME)).execute(task, arg);
    }

    /**
     * Wrap encog model.
     *
     * @param lead Lead.
     */
    private EncogMethodWrapper buildIgniteModel(MLMethodGenome lead) {
        return new EncogMethodWrapper((MLRegression)lead.getPhenotype());
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
