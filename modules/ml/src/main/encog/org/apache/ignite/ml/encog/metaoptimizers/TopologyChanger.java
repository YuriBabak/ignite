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
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.apache.ignite.ml.encog.IgniteNetwork;
import org.apache.ignite.ml.encog.LockKey;
import org.apache.ignite.ml.encog.caches.TrainingContext;
import org.apache.ignite.ml.math.functions.IgniteFunction;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.population.Population;
import org.encog.ml.genetic.MLMethodGeneticAlgorithm;
import org.encog.ml.genetic.MLMethodGenome;

public class TopologyChanger implements Metaoptimizer<TopologyChanger.ScoredTopology, TopologyChanger.Topology> {
    private final IgniteFunction<Integer, Topology> topologySupplier;

    public static class ScoredTopology implements Serializable {
        double score;
        Map<LockKey, Double> weights = new HashMap<>();

        public ScoredTopology(double score, Map<LockKey, Double> weights) {
            this.score = score;
            this.weights = new HashMap<>(weights);
        }
    }

    public static class Topology implements Serializable {
        Map<LockKey, Double> weights = new HashMap<>();

        public Topology() {
            weights = new HashMap<>();
        }

        public Topology(Map<LockKey, Double> m) {
            weights = new HashMap<>(m);
        }
    }

    public TopologyChanger(IgniteFunction<Integer, Topology> initialTopologySupplier) {
        this.topologySupplier = initialTopologySupplier;
    }

    @Override public Topology initialData(int populationNum) {
        return topologySupplier.apply(populationNum);
    }

    @Override public ScoredTopology extractStats(Population population, Topology data,
        TrainingContext ctx) {
        IgniteNetwork phenotype = (IgniteNetwork)(((MLMethodGenome)population.getBestGenome()).getPhenotype());

        return new ScoredTopology(population.getBestGenome().getScore(), phenotype.getLearningMask());
    }

    @Override
    public Map<Integer, Topology> statsAggregator(Map<Integer, ScoredTopology> stats) {
        Set<LockKey> allKeys = new HashSet<>();
        Map<LockKey, Double> resWeights = new HashMap<>();

        stats.values().stream().forEach(st -> {
            allKeys.addAll(st.weights.keySet());
        });

        allKeys.forEach(k -> {
            // Get a weighted average (based on score) of weights of each topology.
            double totl = 0.0;

            for (Integer k1 : stats.keySet())
                totl += stats.get(k1).score;

            if (totl != 0.0) {
                Double weightedAvg = 0.0;
                for (Integer k1 : stats.keySet())
                    weightedAvg += stats.get(k1).weights.get(k) * stats.get(k1).score / totl;

                resWeights.put(k, weightedAvg);
            }
        });

        Map<Integer, Topology> res = new HashMap<>();

        for (Integer subPopulation : stats.keySet())
            res.put(subPopulation, new Topology(resWeights));

        return res;
    }

    @Override
    public MLMethodGeneticAlgorithm statsHandler(MLMethodGeneticAlgorithm train, Topology data) {
        for (Genome genome : train.getGenetic().getPopulation().getSpecies().get(0).getMembers()) {
            IgniteNetwork phenotype = (IgniteNetwork)(((MLMethodGenome)genome).getPhenotype());

            phenotype.reset();

            for (Map.Entry<LockKey, Double> entry : data.weights.entrySet())
                phenotype.lockNeuron(entry.getKey().layer(), entry.getKey().neuron(), entry.getValue());
        }

        return train;
    }
}
