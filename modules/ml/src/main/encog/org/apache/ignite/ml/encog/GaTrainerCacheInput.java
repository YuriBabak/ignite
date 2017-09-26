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

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import javax.cache.Cache;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.encog.caches.TrainingContextCache;
import org.apache.ignite.ml.encog.evolution.replacement.UpdateStrategy;
import org.apache.ignite.ml.math.functions.IgniteSupplier;
import org.encog.ml.MLEncodable;
import org.encog.ml.MLMethod;
import org.encog.ml.MethodFactory;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.ea.opp.EvolutionaryOperator;

public class GaTrainerCacheInput<T extends MLMethod & MLEncodable> implements GATrainerInput<T> {
    private IgniteSupplier<T> mf;
    private String cacheName;
    private int size;
    private int populationSize;

    public GaTrainerCacheInput(String cacheName, IgniteSupplier<T> methodFactory, int size, int populationSize) {
        this.cacheName = cacheName;
        mf = () -> methodFactory.get();
        this.size = size;
        this.populationSize = populationSize;
    }

    @Override public MLDataSet mlDataSet(Ignite ignite) {
        IgniteCache<Integer, MLDataPair> cache = ignite.getOrCreateCache(cacheName);

        System.out.println("Cache size: " + cache.size());

        ArrayList<MLDataPair> lst = new ArrayList<>();

        for (Cache.Entry<Integer, MLDataPair> entry : cache.localEntries())
            lst.add(entry.getValue());

        return new BasicMLDataSet(lst);
    }

    @Override public IgniteSupplier<T> methodFactory() {
        return mf;
    }

    @Override public int datasetSize() {
        return size;
    }

    @Override public int populationSize() {
        return populationSize;
    }

    @Override public List<EvolutionaryOperator> evolutionaryOperators() {
        return null;
    }

    @Override public UpdateStrategy replaceStrategy() {
        return null;
    }
}