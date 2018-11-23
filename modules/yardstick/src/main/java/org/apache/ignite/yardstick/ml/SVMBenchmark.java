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
import org.apache.ignite.IgniteCache;
import org.apache.ignite.cache.affinity.rendezvous.RendezvousAffinityFunction;
import org.apache.ignite.configuration.CacheConfiguration;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.svm.SVMLinearBinaryClassificationTrainer;
import org.apache.ignite.yardstick.cache.IgniteCacheAbstractBenchmark;

/**
 * Ignite benchmark for {@link SVMLinearBinaryClassificationTrainer}
 */
public class SVMBenchmark extends IgniteCacheAbstractBenchmark<Integer, Vector> {
    /** {@inheritDoc} */
    @Override public boolean test(Map<Object, Object> map) throws Exception {
        SVMLinearBinaryClassificationTrainer trainer = new SVMLinearBinaryClassificationTrainer();

        trainer.fit()

        return false;
    }

    @Override protected IgniteCache<Integer, Vector> cache() {
        CacheConfiguration<Integer, double[]> cacheConfiguration = new CacheConfiguration<>();
        cacheConfiguration.setName("YARDSTICK_" + UUID.randomUUID());
        cacheConfiguration.setAffinity(new RendezvousAffinityFunction(false, 10));

        return ignite().getOrCreateCache();
    }
}
