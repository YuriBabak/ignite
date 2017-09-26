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
import org.encog.engine.network.activation.ActivationFunction;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.error.LinearErrorFunction;
import org.encog.neural.flat.FlatNetwork;
import org.encog.neural.networks.ContainsFlat;

public class GradientCalculator {
    private final MLDataSet ds;
    private double[] layerDelta;
    private double[] layerSums;
    private double[] layerOutput;
    private double[] flatSpot;
    private FlatNetwork flat;
    private double[] gradients;
    private double[] weights;

    public GradientCalculator(ContainsFlat network, MLDataSet ds) {
        flat = network.getFlat();
        layerOutput = flat.getLayerOutput();
        layerDelta = new double[layerOutput.length];
        layerSums = flat.getLayerSums();
        flatSpot = new double[flat.getActivationFunctions().length];
        gradients = new double[flat.getWeights().length];
        weights = flat.getWeights();
        this.ds = ds;

        // TODO: at this moment just a hardcode.
        Arrays.fill(flatSpot, 0.1);
    }

    public double[] gradient() {
        // TODO: Here we have rather big overhead with creating many ScaledConjugate gradients, it would be better to just extract
        // gradient calculation from there

        for (MLDataPair d : ds)
            process(d.getInputArray(), d.getIdealArray(), d.getSignificance());

        return gradients;
    }


    private void process(final double[] input, final double[] ideal, double s) {
        double[] actual = new double[flat.getOutputCount()];
        flat.compute(input, actual);

        // TODO: are these functions needed?
        new ErrorCalculation().updateError(actual, ideal, s);
        new LinearErrorFunction().calculateError(ideal, actual, this.layerDelta);

        for (int i = 0; i < actual.length; i++) {
            layerDelta[i] = ((flat.getActivationFunctions()[0]
                .derivativeFunction(this.layerSums[i], this.layerOutput[i]) + this.flatSpot[0]))
                * (layerDelta[i] * s);
        }

        for (int i = flat.getBeginTraining(); i < flat.getEndTraining(); i++)
            processLevel(i);
    }

    private void processLevel(final int currentLevel) {
        final int fromLayerIndex = flat.getLayerIndex()[currentLevel + 1];
        final int toLayerIndex = flat.getLayerIndex()[currentLevel];
        final int fromLayerSize = flat.getLayerCounts()[currentLevel + 1];
        final int toLayerSize = flat.getLayerFeedCounts()[currentLevel];

        final int index = flat.getWeightIndex()[currentLevel];
        final ActivationFunction activation = flat
            .getActivationFunctions()[currentLevel];
        final double currentFlatSpot = this.flatSpot[currentLevel + 1];

        // handle weights
        // array references are made method local to avoid one indirection
        final double[] layerDelta = this.layerDelta;
        final double[] weights = this.weights;
        final double[] gradients = this.gradients;
        final double[] layerOutput = this.layerOutput;
        final double[] layerSums = this.layerSums;

        int yi = fromLayerIndex;
        for (int y = 0; y < fromLayerSize; y++) {
            final double output = layerOutput[yi];
            double sum = 0;

            int wi = index + y;
            final int loopEnd = toLayerIndex+toLayerSize;
            for (int xi = toLayerIndex; xi < loopEnd; xi++, wi += fromLayerSize) {
                gradients[wi] += output * layerDelta[xi];
                sum += weights[wi] * layerDelta[xi];
            }

            layerDelta[yi] = sum
                * (activation.derivativeFunction(layerSums[yi], layerOutput[yi])+currentFlatSpot);

            yi++;
        }
    }

}
