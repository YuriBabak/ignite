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
    public BasicStats extractStats(int subPopulation, Population population, BasicStats data) {
        data.incrTick();
        return data;
    }

    @Override public Map<Integer, BasicStats> statsAggregator(
        Map<Integer, BasicStats> stats) {
        double globalBest = stats.values().stream().mapToDouble(BasicStats::bestScore).min().orElse(Double.POSITIVE_INFINITY);
        int tick = stats.get(0).tick;
        long prevTickTime = stats.values().stream().mapToLong(BasicStats::prevGlobalTickTime).min().getAsLong();
        long time = System.currentTimeMillis();

        System.out.println("");

        BasicStats data = new BasicStats(globalBest, tick, time, time - prevTickTime);

        Map<Integer, BasicStats> res = new HashMap<>();

        stats.keySet().forEach(k -> res.put(k, data));

        return res;
    }

    @Override
    public MLMethodGeneticAlgorithm statsHandler(int subPopulation, MLMethodGeneticAlgorithm train, BasicStats data) {
        return train;
    }

    public static class BasicStats implements Serializable {
        long prevGlobalTickTime;
        long currentGlobalTickDuration;

        double bestScore;
        int tick;

        public BasicStats(double bestScore, int tick) {
            this.bestScore = bestScore;
            this.tick = tick;
            prevGlobalTickTime = System.currentTimeMillis();
        }

        public BasicStats(double bestScore, int tick, long prevTickTime, long currentGlobalTickDuration) {
            this.bestScore = bestScore;
            this.tick = tick;
            prevGlobalTickTime = prevTickTime;
            this.currentGlobalTickDuration = currentGlobalTickDuration;
        }

        public double bestScore() {
            return bestScore;
        }

        public int tick() {
            return tick;
        }

        public void incrTick() {
            tick++;
        }

        public long prevGlobalTickTime() {
            return prevGlobalTickTime;
        }

        public long currentGlobalTickDuration() {
            return currentGlobalTickDuration;
        }
    }
}
