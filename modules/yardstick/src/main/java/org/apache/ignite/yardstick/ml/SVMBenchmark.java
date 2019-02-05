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
import java.util.Random;
import java.util.UUID;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.cache.affinity.rendezvous.RendezvousAffinityFunction;
import org.apache.ignite.configuration.CacheConfiguration;
import org.apache.ignite.ml.IgniteModel;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.math.primitives.vector.VectorUtils;
import org.apache.ignite.ml.selection.scoring.evaluator.BinaryClassificationEvaluator;
import org.apache.ignite.ml.svm.SVMLinearClassificationModel;
import org.apache.ignite.ml.svm.SVMLinearClassificationTrainer;
import org.apache.ignite.yardstick.cache.IgniteCacheAbstractBenchmark;
import org.yardstickframework.BenchmarkConfiguration;
import org.yardstickframework.BenchmarkUtils;

/**
 * Ignite benchmark for {@link SVMLinearClassificationTrainer}
 */
public class SVMBenchmark extends IgniteCacheAbstractBenchmark<Integer, Vector> {
    /** Dimension. */
    private static int DIMENSION = 2;
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
    IgniteCache<Integer, Vector> cache;

    /** {@inheritDoc} */
    @Override public boolean test(Map<Object, Object> map) throws Exception {
        SVMLinearClassificationTrainer trainer = new SVMLinearClassificationTrainer();

        Ignite ignite = ignite();
        cache = cache();

        assert ignite != null;
        assert cache != null;

        
        featureExtractor = (IgniteBiFunction<Integer, Vector, Vector>)(integer, vector) -> vector.copyOfRange(1, vector.size());
        lbExtractor = (IgniteBiFunction<Integer, Vector, Double>)(integer, vector) -> vector.get(0);

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
    @Override protected IgniteCache<Integer, Vector> cache() {
        CacheConfiguration<Integer, Vector> cacheConfiguration = new CacheConfiguration<>();
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
        IgniteCache<Integer, Vector> cache = cache();
        Random random = new Random(SEED);

        for (int i = 0; i < DATASET_SIZE; i++)
            cache.put(i, nextPoint(random));
    }

    /** Generate a next random point. */
    private Vector nextPoint(Random r){
        double lb = r.nextBoolean() ? 1d : 0d;
        double[] pnt = new double[DIMENSION + 1];

        pnt[0] = lb;

        for (int i = 1; i < DIMENSION + 1; i++)
            pnt[i] = lb - r.nextDouble();

        return VectorUtils.of(pnt);
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

        BenchmarkUtils.println("\n>>> Accuracy + " + accuracy);
    }
}
