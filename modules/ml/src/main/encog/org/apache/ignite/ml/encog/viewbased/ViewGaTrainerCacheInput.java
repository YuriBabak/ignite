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

package org.apache.ignite.ml.encog.viewbased;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import javax.cache.Cache;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.cluster.ClusterNode;
import org.apache.ignite.lang.IgnitePredicate;
import org.apache.ignite.ml.encog.GATrainerInput;
import org.apache.ignite.ml.encog.evolution.operators.IgniteEvolutionaryOperator;
import org.apache.ignite.ml.encog.metaoptimizers.Metaoptimizer;
import org.apache.ignite.ml.encog.util.Util;
import org.apache.ignite.ml.math.distributed.CacheUtils;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.apache.ignite.ml.math.functions.IgniteFunction;
import org.apache.ignite.ml.math.functions.IgniteSupplier;
import org.encog.ml.CalculateScore;
import org.encog.ml.MLEncodable;
import org.encog.ml.MLMethod;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;

public class ViewGaTrainerCacheInput<T extends MLMethod & MLEncodable, S, U extends Serializable> implements GATrainerInput<T, S, U> {
    private final IgniteBiFunction<GATrainerInput, Ignite, CalculateScore> scoreCalculatorSupplier;
    private final int speciesCount;
    private final IgnitePredicate<Map<Integer, U>> stopCriterion;
    private final int historyDepth;
    private IgniteFunction<Integer, T> mf;
    private String cacheName;
    private int size;
    private int populationSize;
    private List<IgniteEvolutionaryOperator> evolutionaryOperators;
    private int iterationsPerLocalTick;
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
    public ViewGaTrainerCacheInput(String cacheName,
        IgniteFunction<Integer, T> mtdFactory,
        int size,
        int populationSize,
        List<IgniteEvolutionaryOperator> evolutionaryOperators,
        int iterationsPerLocalTick,
        IgniteBiFunction<GATrainerInput, Ignite, CalculateScore> scoreCalculator,
        int speciesCount,
        Metaoptimizer<S, U> metaoptimizer,
        double batchPercentage,
        IgnitePredicate<Map<Integer, U>> stopCriterion,
        int historyDepth) {
        this.cacheName = cacheName;
        mf = mtdFactory;
        this.size = size;
        this.populationSize = populationSize;
        this.evolutionaryOperators = evolutionaryOperators;
        this.iterationsPerLocalTick = iterationsPerLocalTick;
        this.scoreCalculatorSupplier = scoreCalculator;
        this.speciesCount = speciesCount;
        this.metaoptimizer = metaoptimizer;
        this.batchPercentage = batchPercentage;
        this.stopCriterion = stopCriterion;
        this.historyDepth = historyDepth;
    }

    @Override public MLDataSet mlDataSet(int subPop, Ignite ignite) {
        IgniteCache<Integer, double[]> cache = ignite.getOrCreateCache(cacheName);
        ClusterNode localNode = ignite.cluster().localNode();
        List<Integer> localKeys = IntStream.range(0, datasetSize()).boxed().filter(i -> ignite.affinity(cacheName).mapKeyToNode(i).equals(localNode)).collect(Collectors.toList());

        int totalKeys = localKeys.size();
        System.out.println("Local keys size: " + totalKeys);

        LinkedList<MLDataPair> lst = new LinkedList<>();
        for (Cache.Entry<Integer, double[]> entry : cache.localEntries()) {
            double[] arr = entry.getValue();
            int offsetLimit = (arr.length - historyDepth - 1);
            int[] offsets = Util.selectKDistinct(offsetLimit, (int)((offsetLimit + 1) * batchPercentage));
            for (int offset : offsets)
                lst.add(new BasicMLDataPair(new MLDataView(arr, offset, historyDepth), new MLDataView(arr, offset + historyDepth, 1)));
        }

        return new BasicMLDataSet(lst);
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

    @Override public int iterationsPerLocalTick() {
        return iterationsPerLocalTick;
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