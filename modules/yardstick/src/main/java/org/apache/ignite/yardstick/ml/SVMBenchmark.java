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

import java.util.Map;
import java.util.UUID;
import java.util.stream.Stream;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.cache.affinity.rendezvous.RendezvousAffinityFunction;
import org.apache.ignite.configuration.CacheConfiguration;
import org.apache.ignite.ml.IgniteModel;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.selection.scoring.evaluator.BinaryClassificationEvaluator;
import org.apache.ignite.ml.structures.LabeledVector;
import org.apache.ignite.ml.svm.SVMLinearClassificationModel;
import org.apache.ignite.ml.svm.SVMLinearClassificationTrainer;
import org.apache.ignite.ml.util.generators.standard.TwoSeparableClassesDataStream;
import org.apache.ignite.yardstick.cache.IgniteCacheAbstractBenchmark;
import org.yardstickframework.BenchmarkConfiguration;
import org.yardstickframework.BenchmarkUtils;

/**
 * Ignite benchmark for {@link SVMLinearClassificationTrainer}
 */
public class SVMBenchmark extends IgniteCacheAbstractBenchmark<Integer, Vector> {
    /** Dataset size. */
    private static int DATASET_SIZE = 30000;
    /** Seed. */
    private static long SEED = 123456L;

    /** Model. */
    SVMLinearClassificationModel model;
    /** Feature extractor. */
    IgniteBiFunction featureExtractor;
    /** Label extractor. */
    IgniteBiFunction lbExtractor;
    /** Cache. */
    IgniteCache<Integer, LabeledVector<Double>> cache;

    /** {@inheritDoc} */
    @Override public boolean test(Map<Object, Object> map) throws Exception {
        SVMLinearClassificationTrainer trainer = new SVMLinearClassificationTrainer();

        Ignite ignite = ignite();
        cache = cache();

        assert ignite != null;
        assert cache != null;

        
        featureExtractor = (IgniteBiFunction<Integer, LabeledVector<Double>, Vector>)(integer, vector) -> vector.features();
        lbExtractor = (IgniteBiFunction<Integer, LabeledVector<Double>, Double>)(integer, vector) -> (Double) vector.label();

        model = trainer.fit(
            ignite,
            cache,
            featureExtractor,
            lbExtractor
        );

        return false;
    }

    /** {@inheritDoc} */
    @Override public void tearDown() throws Exception {
        modelEvaluation(cache,model,featureExtractor,lbExtractor);

        super.tearDown();
    }

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

    /** Fill cache by random points. */
    private void fillCache() {
        IgniteCache<Integer, LabeledVector<Double>> cache = cache();

        Stream<LabeledVector<Double>> dataStream = new TwoSeparableClassesDataStream(0, 20, SEED).labeled();

        dataStream.limit(DATASET_SIZE).forEach(vector -> {
            cache.put(vector.hashCode(), vector);
        });

    }

    /** Evaluate trained model. */
    private void modelEvaluation(IgniteCache dataCache, IgniteModel<Vector, Double> mdl,
        IgniteBiFunction featureExtractor, IgniteBiFunction lbExtractor){
        double accuracy = BinaryClassificationEvaluator.evaluate(
            dataCache,
            mdl,
            featureExtractor,
            lbExtractor
        ).accuracy();

        BenchmarkUtils.println(">>> Accuracy: " + accuracy);
    }
}
