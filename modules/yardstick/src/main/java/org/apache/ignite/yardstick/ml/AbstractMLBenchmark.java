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

package org.apache.ignite.yardstick.ml;

import java.util.UUID;
import java.util.stream.Stream;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.cache.affinity.rendezvous.RendezvousAffinityFunction;
import org.apache.ignite.configuration.CacheConfiguration;
import org.apache.ignite.ml.IgniteModel;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.selection.scoring.evaluator.BinaryClassificationEvaluator;
import org.apache.ignite.ml.structures.LabeledVector;
import org.apache.ignite.ml.util.generators.standard.TwoSeparableClassesDataStream;
import org.apache.ignite.yardstick.cache.IgniteCacheAbstractBenchmark;
import org.yardstickframework.BenchmarkConfiguration;
import org.yardstickframework.BenchmarkUtils;

/**
 * Abstract yardstick benchmark for binary classification.
 */
public abstract class AbstractMLBenchmark extends IgniteCacheAbstractBenchmark<Integer, Vector> {
    /** Seed. */
    private static long SEED = 123456L;

    /** {@inheritDoc} */
    @Override protected IgniteCache cache() {
        CacheConfiguration<UUID, LabeledVector<Double>> cacheConfiguration = new CacheConfiguration<>();
        cacheConfiguration.setName("YARDSTICK_" + UUID.randomUUID());
        cacheConfiguration.setAffinity(new RendezvousAffinityFunction(false, 10));

        return ignite().getOrCreateCache(cacheConfiguration);
    }

    /** {@inheritDoc} */
    @Override public void setUp(BenchmarkConfiguration cfg) throws Exception {
        super.setUp(cfg);
        fillCache();
    }

    /** Evaluate trained model. */
    public void modelEvaluation(IgniteCache dataCache, IgniteModel<Vector, Double> mdl,
        IgniteBiFunction featureExtractor, IgniteBiFunction lbExtractor){
        double accuracy = BinaryClassificationEvaluator.evaluate(
            dataCache,
            mdl,
            featureExtractor,
            lbExtractor
        ).accuracy();

        BenchmarkUtils.println(">>> Accuracy: " + accuracy);
    }

    public abstract int getDataSetSize();

    /** Fill cache by random points. */
    private void fillCache() {
        IgniteCache<Integer, LabeledVector<Double>> cache = cache();

        Stream<LabeledVector<Double>> dataStream = new TwoSeparableClassesDataStream(0, 20, SEED).labeled();

        dataStream.limit(getDataSetSize()).forEach(vector -> {
            cache.put(vector.hashCode(), vector);
        });

    }
}
