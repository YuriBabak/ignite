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
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.cluster.ClusterNode;
import org.apache.ignite.lang.IgnitePredicate;
import org.apache.ignite.ml.encog.evolution.operators.IgniteEvolutionaryOperator;
import org.apache.ignite.ml.encog.metaoptimizers.Metaoptimizer;
import org.apache.ignite.ml.encog.util.Util;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.apache.ignite.ml.math.functions.IgniteFunction;
import org.apache.ignite.ml.math.functions.IgniteSupplier;
import org.encog.ml.CalculateScore;
import org.encog.ml.MLEncodable;
import org.encog.ml.MLMethod;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;

public class GaTrainerCacheInput<T extends MLMethod & MLEncodable, S, U extends Serializable> implements GATrainerInput<T, S, U> {
    private final IgniteBiFunction<GATrainerInput, Ignite, CalculateScore> scoreCalculatorSupplier;
    private final int speciesCount;
    private final IgnitePredicate<Map<Integer, U>> stopCriterion;
    private final IgniteFunction<Integer, Integer> localTickCountStrategy;
    private IgniteFunction<Integer, T> mf;
    private String cacheName;
    private int size;
    private int populationSize;
    private List<IgniteEvolutionaryOperator> evolutionaryOperators;
    private Metaoptimizer<S, U> metaoptimizer;
    private double batchPercentage;

    /**
     *
     * @param cacheName
     * @param mtdFactory
     * @param size Size of training set
     * @param populationSize
     * @param evolutionaryOperators
     * @param iterationsPerLocalTick
     * @param scoreCalculator
     * @param speciesCount
     * @param metaoptimizer
     * @param batchPercentage
     */
    public GaTrainerCacheInput(String cacheName,
        IgniteFunction<Integer, T> mtdFactory,
        int size,
        int populationSize,
        List<IgniteEvolutionaryOperator> evolutionaryOperators,
        IgniteFunction<Integer, Integer> localTickCountStrategy,
        IgniteBiFunction<GATrainerInput, Ignite, CalculateScore> scoreCalculator,
        int speciesCount,
        Metaoptimizer<S, U> metaoptimizer,
        double batchPercentage,
        IgnitePredicate<Map<Integer, U>> stopCriterion) {
        this.cacheName = cacheName;
        mf = mtdFactory;
        this.size = size;
        this.populationSize = populationSize;
        this.evolutionaryOperators = evolutionaryOperators;
        this.localTickCountStrategy = localTickCountStrategy;
        this.scoreCalculatorSupplier = scoreCalculator;
        this.speciesCount = speciesCount;
        this.metaoptimizer = metaoptimizer;
        this.batchPercentage = batchPercentage;
        this.stopCriterion = stopCriterion;
    }

    @Override public MLDataSet mlDataSet(int subPop, Ignite ignite) {
        IgniteCache<Integer, MLDataPair> cache = ignite.getOrCreateCache(cacheName);
        System.out.println("Total entries in dataset (cache " + cache.getName() + "): " + cache.size());

//        System.out.println("dataset cache size: " + cache.size());

        ArrayList<MLDataPair> lst = new ArrayList<>();

        ClusterNode locNode = ignite.cluster().localNode();
        List<Integer> locKeys = IntStream.range(0, datasetSize()).boxed().filter(i -> ignite.affinity(cacheName).mapKeyToNode(i).equals(locNode)).collect(Collectors.toList());

        int totalKeys = locKeys.size();
        System.out.println("Local keys size: " + totalKeys);
        int subsetSize = (int)(totalKeys * batchPercentage);

        int[] subset = Util.selectKDistinct(totalKeys, subsetSize);

        System.out.println("Subset size: " + subset.length);

        for (int i : subset) {
            MLDataPair pair = cache.get(locKeys.get(i));
            lst.add(pair);
        }

        BasicMLDataSet res = new BasicMLDataSet(lst);
        System.out.println("Generated dataset of size " + res.size() + " dimensions: " + res.get(0).getInput().size() + "," + res.get(0).getIdeal().size());
        return res;
    }

    @Override public IgniteSupplier<T> methodFactory(int i) {
        return () -> mf.apply(i);
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

    @Override public int iterationsPerLocalTick(int subPopulation) {
        return localTickCountStrategy.apply(subPopulation);
    }

    @Override public CalculateScore scoreCalculator(Ignite ignite) {
        return scoreCalculatorSupplier.apply(this, ignite);
    }

    @Override public int subPopulationsCount() {
        return speciesCount;
    }

    @Override public Metaoptimizer<S, U> metaoptimizer() {
        return metaoptimizer;
    }

    @Override public boolean shouldStop(Map<Integer, U> data) {
        return stopCriterion.apply(data);
    }
}