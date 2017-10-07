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
import java.util.Map;
import org.encog.ml.ea.population.Population;
import org.encog.ml.genetic.MLMethodGeneticAlgorithm;

public class BasicStatsCounter implements Metaoptimizer<BasicStatsCounter.BasicStats, BasicStatsCounter.BasicStats> {
    @Override public BasicStats initialData(int populationNum) {
        return new BasicStats(Double.POSITIVE_INFINITY, 0);
    }

    @Override
    public BasicStats extractStats(Population population, BasicStats data) {
        return new BasicStats(population.getBestGenome().getScore(), data.tick + 1);
    }

    @Override public Map<Integer, BasicStats> statsAggregator(
        Map<Integer, BasicStats> stats) {
        double globalBest = stats.values().stream().mapToDouble(BasicStats::bestScore).min().orElse(Double.POSITIVE_INFINITY);
        int tick = stats.get(0).tick;

        BasicStats data = new BasicStats(globalBest, tick);

        Map<Integer, BasicStats> res = new HashMap<>();

        stats.keySet().forEach(k -> {
            res.put(k, data);
        });

        return res;
    }

    @Override
    public MLMethodGeneticAlgorithm statsHandler(MLMethodGeneticAlgorithm train, BasicStats data) {
        return train;
    }

    public static class BasicStats implements Serializable {
        double bestScore;
        int tick;

        public BasicStats(double bestScore, int tick) {
            this.bestScore = bestScore;
            this.tick = tick;
        }

        public double bestScore() {
            return bestScore;
        }

        public int tick() {
            return tick;
        }
    }
}
