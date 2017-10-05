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

package org.apache.ignite.ml.encog.evolution.operators;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.encog.IgniteNetwork;
import org.apache.ignite.ml.encog.util.Util;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.train.EvolutionaryAlgorithm;
import org.encog.ml.genetic.MLMethodGenome;
import org.encog.neural.networks.BasicNetwork;

public class CrossoverFeatures extends IgniteEvolutionaryOperator {

    private MLDataSet ds;

    public CrossoverFeatures(double prob, String operatorId) {
        super(prob, operatorId);
    }

    @Override public void init(EvolutionaryAlgorithm theOwner) {
        super.init(theOwner);
        ds = input().mlDataSet(ignite());
    }

    @Override public int offspringProduced() {
        return 1;
    }

    @Override public int parentsNeeded() {
        return 2;
    }

    @Override public void performOperation(Random rnd, Genome[] parents, int parentIndex, Genome[] offspring,
        int offspringIndex) {

        IgniteNetwork p1 = (IgniteNetwork)((MLMethodGenome)parents[0]).getPhenotype();
        BasicNetwork p2 = (IgniteNetwork)((MLMethodGenome)parents[1]).getPhenotype();

        BasicNetwork off = (BasicNetwork)p2.clone();

        int[] sampleData = Util.selectKDistinct(ds.size(), 20);

        Map<Integer, List<Integer>> neuron2Pos1 = initNeuron2Pos(p1);
        Map<Integer, List<Integer>> neuron2Pos2 = initNeuron2Pos(p2);

        for (int datum : sampleData) {
            MLDataPair pair = ds.get(datum);
            p1.compute(pair.getInput());
            p2.compute(pair.getInput());

            IgniteBiTuple<Integer, Double>[] neuron2Activation1 = new IgniteBiTuple[p1.getLayerNeuronCount(1)];
            IgniteBiTuple<Integer, Double>[] neuron2Activation2 = new IgniteBiTuple[p2.getLayerNeuronCount(1)];

            fillArr(p1, neuron2Activation1);
            fillArr(p2, neuron2Activation2);

            updateNeuron2Pos(neuron2Pos1, neuron2Activation1);
            updateNeuron2Pos(neuron2Pos2, neuron2Activation2);
        }

        Integer[] feature2Neuron1 = feature2Neuron(neuron2Pos1);
        Integer[] feature2Neuron2 = feature2Neuron(neuron2Pos2);

        for (int feature = 0; feature < feature2Neuron1.length; feature++) {
            int canonicalNeuron = feature2Neuron1[feature];
            int selfNeuron = feature2Neuron2[feature];

            normalize(p2, off, canonicalNeuron, selfNeuron);
        }

        offspring[offspringIndex] = new MLMethodGenome(off);
    }

    private void normalize(BasicNetwork p2, BasicNetwork off, int canonicalNeuron, int selfNeuron) {
        for (int i = 0; i < p2.getLayerNeuronCount(0); i++)
            off.setWeight(0, i, canonicalNeuron, p2.getWeight(0, i, selfNeuron));

        for (int i = 0; i < p2.getLayerNeuronCount(2); i++)
            off.setWeight(1, canonicalNeuron, i, p2.getWeight(1, selfNeuron, i));
    }

    private void updateNeuron2Pos(Map<Integer, List<Integer>> n2Pos, IgniteBiTuple<Integer, Double>[] n2a) {
        for (int i = 0; i < n2a.length; i++) {
            IgniteBiTuple<Integer, Double> posAndAct = n2a[i];

            n2Pos.get(posAndAct.get1()).add(i);
        }
    }

    private Map<Integer, List<Integer>> initNeuron2Pos(BasicNetwork n) {
        Map<Integer, List<Integer>> res = new HashMap<>();

        for (int i = 0; i < n.getLayerNeuronCount(1); i++)
            res.put(i, new LinkedList<>());

        return res;
    }

    private void fillArr(BasicNetwork n, IgniteBiTuple<Integer, Double>[] arr) {
        for (int i = 0; i < n.getLayerNeuronCount(1); i++)
            arr[i] = new IgniteBiTuple<>(i, n.getLayerOutput(1, i));

        Arrays.sort(arr, Comparator.comparingDouble(IgniteBiTuple::get2));
    }

    private Integer[] feature2Neuron(Map<Integer, List<Integer>> m) {
        // Get average positions for each neuron.
        List<IgniteBiTuple<Integer, Integer>> res = m.entrySet().stream().map(e -> new IgniteBiTuple<>(e.getKey(), (int)e.getValue().stream().mapToInt(i -> i).average().getAsDouble())).collect(Collectors.toList());
        res.sort(Comparator.comparingDouble(IgniteBiTuple::get2));
        // Now positions order is treated as features order.
        return res.stream().map(IgniteBiTuple::get1).collect(Collectors.toList()).toArray(new Integer[] {});
    }
}
