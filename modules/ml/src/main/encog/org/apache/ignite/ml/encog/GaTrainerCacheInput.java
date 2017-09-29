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
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import javax.cache.Cache;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.cluster.ClusterNode;
import org.apache.ignite.ml.encog.caches.TrainingContext;
import org.apache.ignite.ml.encog.evolution.operators.IgniteEvolutionaryOperator;
import org.apache.ignite.ml.encog.metaoptimizers.Metaoptimizer;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.apache.ignite.ml.math.functions.IgniteSupplier;
import org.encog.ml.CalculateScore;
import org.encog.ml.MLEncodable;
import org.encog.ml.MLMethod;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;

public class GaTrainerCacheInput<T extends MLMethod & MLEncodable, S, U extends Serializable> implements GATrainerInput<T, S, U> {
    private final IgniteBiFunction<TrainingContext, Ignite, CalculateScore> scoreCalculatorSupplier;
    private final int speciesCount;
    private IgniteSupplier<T> mf;
    private String cacheName;
    private int size;
    private int populationSize;
    private List<IgniteEvolutionaryOperator> evolutionaryOperators;
    private int iterationsPerLocalTick;
    private Metaoptimizer<S, U> metaoptimizer;
    private double batchPercentage;

    public GaTrainerCacheInput(String cacheName,
        IgniteSupplier<T> mtdFactory,
        int size,
        int populationSize,
        List<IgniteEvolutionaryOperator> evolutionaryOperators,
        int iterationsPerLocalTick,
        IgniteBiFunction<TrainingContext, Ignite, CalculateScore> scoreCalculator,
        int speciesCount,
        Metaoptimizer<S, U> metaoptimizer,
        double batchPercentage) {
        this.cacheName = cacheName;
        mf = () -> mtdFactory.get();
        this.size = size;
        this.populationSize = populationSize;
        this.evolutionaryOperators = evolutionaryOperators;
        this.iterationsPerLocalTick = iterationsPerLocalTick;
        this.scoreCalculatorSupplier = scoreCalculator;
        this.speciesCount = speciesCount;
        this.metaoptimizer = metaoptimizer;
        this.batchPercentage = batchPercentage;
    }

    @Override public MLDataSet mlDataSet(Ignite ignite) {
        IgniteCache<Integer, MLDataPair> cache = ignite.getOrCreateCache(cacheName);

//        System.out.println("dataset cache size: " + cache.size());

        ArrayList<MLDataPair> lst = new ArrayList<>();

        ClusterNode localNode = ignite.cluster().localNode();

        List<Integer> localKeys = IntStream.range(0, datasetSize()).boxed().filter(i -> ignite.affinity(cacheName).mapKeyToNode(i).equals(localNode)).collect(Collectors.toList());

        int totalKeys = localKeys.size();
        int subsetSize = (int)(totalKeys * batchPercentage);

        int[] subset = Util.selectKDistinct(totalKeys, subsetSize);

        for (int i : subset)
            lst.add(cache.get(localKeys.get(i)));

//
//        for (Cache.Entry<Integer, MLDataPair> entry : cache.localEntries())
//            lst.add(entry.getValue());

        return new BasicMLDataSet(lst);
    }

    @Override public IgniteSupplier<T> methodFactory() {
        return mf;
    }

    @Override public int datasetSize() {
        return size;
    }

    @Override public int subPopulationSize() {
        return populationSize;
    }

    @Override public List<IgniteEvolutionaryOperator> evolutionaryOperators() {
        return evolutionaryOperators;
    }

    @Override public int iterationsPerLocalTick() {
        return iterationsPerLocalTick;
    }

    @Override public CalculateScore scoreCalculator(TrainingContext ctx, Ignite ignite) {
        return scoreCalculatorSupplier.apply(ctx, ignite);
    }

    @Override public int subPopulationsCount() {
        return speciesCount;
    }

    @Override public Metaoptimizer<S, U> metaoptimizer() {
        return metaoptimizer;
    }
}