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

package org.apache.ignite.ml.encog.metaoptimizers;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.encog.util.StreamUtils;
import org.apache.ignite.ml.encog.caches.TrainingContext;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.encog.ml.ea.population.Population;
import org.encog.ml.genetic.MLMethodGeneticAlgorithm;
import org.encog.ml.genetic.MLMethodGenome;

public interface Metaoptimizer<S, U extends Serializable> extends Serializable {
    S extractStats(Population population, U data, TrainingContext ctx);

    // At i-th position of the list contained data which should be sent to i-th iteration.
    Map<Integer, U> statsAggregator(Map<Integer, S> stats);

    MLMethodGeneticAlgorithm statsHandler(MLMethodGeneticAlgorithm train, U data);

    /**
     * Convinient way to combine multiple metaoptimizers in a chain.
     * @param op
     * @param <S1>
     * @param <U1>
     * @return
     */
    default <S1, U1 extends Serializable> Metaoptimizer<IgniteBiTuple<S, S1>, IgniteBiTuple<U, U1>> andThen(Metaoptimizer<S1, U1> op) {
        Metaoptimizer<S, U> outerThis = this;
        return new Metaoptimizer<IgniteBiTuple<S, S1>, IgniteBiTuple<U, U1>>() {
            @Override public IgniteBiTuple<S, S1> extractStats(Population population, IgniteBiTuple<U, U1> data, TrainingContext ctx) {
                return new IgniteBiTuple<>(outerThis.extractStats(population, Optional.ofNullable(data).map(IgniteBiTuple::get1).orElse(null), ctx), op.extractStats(population, Optional.ofNullable(data).map(IgniteBiTuple::get2).orElse(null), ctx));
            }

            @Override public Map<Integer, IgniteBiTuple<U, U1>> statsAggregator(Map<Integer, IgniteBiTuple<S, S1>> stats) {
                Map<Integer, IgniteBiTuple<U, U1>> res = new HashMap<>();

                Map<Integer, S> m1 = new HashMap<>();
                for (Map.Entry<Integer, IgniteBiTuple<S, S1>> entry : stats.entrySet()) {
                    Integer subPopulation = entry.getKey();
                    IgniteBiTuple<S, S1> data = entry.getValue();

                    m1.put(subPopulation, data.get1());
                }

                Map<Integer, S1> m2 = new HashMap<>();
                for (Map.Entry<Integer, IgniteBiTuple<S, S1>> entry : stats.entrySet()) {
                    Integer subPopulation = entry.getKey();
                    IgniteBiTuple<S, S1> data = entry.getValue();

                    m2.put(subPopulation, data.get2());
                }

                Map<Integer, U> r1 = outerThis.statsAggregator(m1);
                Map<Integer, U1> r2 = op.statsAggregator(m2);

                for (Integer subPopulation : r1.keySet())
                    res.put(subPopulation, new IgniteBiTuple<>(r1.get(subPopulation), r2.get(subPopulation)));

                return res;
            }

            @Override
            public MLMethodGeneticAlgorithm statsHandler(MLMethodGeneticAlgorithm train, IgniteBiTuple<U, U1> data) {
                return op.statsHandler(outerThis.statsHandler(train, data.get1()), data.get2());
            }
        };
    }
}
