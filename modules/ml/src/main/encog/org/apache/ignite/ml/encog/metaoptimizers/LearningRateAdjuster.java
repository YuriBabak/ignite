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
import org.apache.ignite.ml.encog.caches.TrainingContext;
import org.apache.ignite.ml.encog.evolution.operators.HasLearningRate;
import org.apache.ignite.ml.encog.util.TrainingUtils;
import org.encog.ml.ea.population.Population;
import org.encog.ml.genetic.MLMethodGeneticAlgorithm;

/**
 * Implementation of {@link Metaoptimizer} for adjustable learning(mutation) rate.
 */
public class LearningRateAdjuster implements Metaoptimizer<LearningRateAdjuster.LearningRateStats, LearningRateAdjuster.LearningRateStats> {
    public static class LearningRateStats implements Serializable {
        Double score;
        Double relImprovement;
        Double learningRate;

        public LearningRateStats(Double score, Double relImprovement, Double learningRate) {
            this.score = score;
            this.relImprovement = relImprovement;
            this.learningRate = learningRate;
        }

        @Override public String toString() {
            return "LearningRateStats [" +
                "score=" + score +
                ", relImprovement=" + relImprovement +
                ", learningRate=" + learningRate +
                ']';
        }
    }

    @Override public LearningRateStats initialData(int populationNum) {
        return null;
    }

    @Override public LearningRateStats extractStats(Population population, LearningRateStats prevStats, TrainingContext ctx) {
        double curScure = population.getBestGenome().getScore();
        if (prevStats == null)
            return new LearningRateStats(curScure, null, 1.0);
        else {
            double improvement = (prevStats.score - curScure);
            return new LearningRateStats(curScure, improvement, prevStats.learningRate * (1 + improvement));
        }
    }

    @Override public Map<Integer, LearningRateStats> statsAggregator(Map<Integer, LearningRateStats> stats) {
        System.out.println(stats);
        return stats;
    }

    @Override public MLMethodGeneticAlgorithm statsHandler(MLMethodGeneticAlgorithm train, LearningRateStats stats) {
        TrainingUtils.getOperatorsByClass(train, HasLearningRate.class).forEach(op -> {
            op.setLearningRate(10 * stats.relImprovement);
        });

        return train;
    }
}
