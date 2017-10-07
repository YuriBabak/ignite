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
import java.util.Map;
import org.apache.ignite.ml.math.functions.IgniteFunction;
import org.apache.ignite.ml.math.statistics.Variance;
import org.encog.ml.ea.population.Population;
import org.encog.ml.genetic.MLMethodGeneticAlgorithm;

/**
 * Implementation of {@link Metaoptimizer} for adjustable learning(mutation) rate.
 */
public class LearningRateAdjuster implements Metaoptimizer<LearningRateAdjuster.LearningRateStats, LearningRateAdjuster.LearningRateStats> {
    private int maxGlobalTicks;

    private IgniteFunction<Integer, Double> learningRateProvider;
    private int subPopulations;

    public LearningRateAdjuster(
        IgniteFunction<Integer, Double> learningRateProvider, int subPopulations) {
        this.learningRateProvider = learningRateProvider;
        this.subPopulations = subPopulations;
    }

    public static class LearningRateStats implements Serializable {
        Variance scoresVar;
        Double learningRate;
        Double weightedAverageLearningRate;

        public LearningRateStats(Double learningRate) {
            this.learningRate = learningRate;
            this.
            scoresVar = new Variance();
        }

        public LearningRateStats update(double score) {
            this.scoresVar.update(score);
            return this;
        }
    }

    @Override public LearningRateStats initialData(int populationNum) {
        return new LearningRateStats(0.1);
    }

    @Override public LearningRateStats extractStats(Population population, LearningRateStats prevStats) {
        double curScore = population.getBestGenome().getScore();
        return prevStats.update(curScore);
    }

    @Override public Map<Integer, LearningRateStats> statsAggregator(Map<Integer, LearningRateStats> stats) {
        System.out.println(stats);
        return stats;
    }

    @Override public MLMethodGeneticAlgorithm statsHandler(MLMethodGeneticAlgorithm train, LearningRateStats stats) {
        double dispIdx = stats.scoresVar.m2() / stats.scoresVar.mean();

        System.out.println("Disp idx " + dispIdx);

        return train;
    }
}
