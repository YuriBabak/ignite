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

package org.apache.ignite.ml.encog;

import java.io.Serializable;
import java.util.Objects;
import java.util.Random;
import org.encog.mathutil.BoundMath;
import org.encog.ml.MLEncodable;
import org.encog.ml.MLMethod;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.util.obj.ObjectCloner;

public class TreeNetwork implements MLEncodable, MLMethod, MLRegression, Serializable {
    double[] weights;
    double[] compute;
    int depth;
    int inputCount;

    public TreeNetwork(int depth) {
        this.depth = depth;
        inputCount = (int)Math.pow(2, depth - 1);
        weights = new double[(int)(Math.pow(2, depth) - 2)];
        compute = new double[(int)(Math.pow(2, depth) - 1)];

        randomize();
    }

    private void randomize() {
        Random random = new Random();
        for (int i = 0; i < weights.length; i++)
            weights[i] = (random.nextDouble() - 0.5) * 2;

    }

    @Override public int encodedArrayLength() {
        return weights.length;
    }

    // Ignore to neuron because it is always determined. We leave this api for the ease of compatibility.
    public void setWeight(int fromLayer, int fromNeuron, int toNeuron, double val) {
        int offsetInput = compute.length - ((int)Math.pow(2, depth - fromLayer) - 1);
        weights[offsetInput + fromNeuron] = val;
    }

    public double getWeight(int fromLayer, int fromNeuron, int toNeuron) {
        int offsetInput = compute.length - ((int)Math.pow(2, depth - fromLayer) - 1);
        return weights[offsetInput + fromNeuron];
    }

    public int getLayerNeuronCount(int l) {
        return (int)Math.pow(2, depth - l - 1);
    }

    @Override public void encodeToArray(double[] encoded) {
        System.arraycopy(weights, 0, encoded, 0, weights.length);
    }

    @Override public void decodeFromArray(double[] encoded) {
        System.arraycopy(encoded, 0, weights, 0, encoded.length);
    }

    @Override public MLData compute(MLData input) {
        System.arraycopy(input.getData(), 0, compute, 0, input.size());

        for (int l = 0; l < depth - 1; l++) {
            int neuronsInLayer = (int)Math.pow(2, depth - l - 1);
            int offsetInput = compute.length - ((int)Math.pow(2, depth - l) - 1);
            int offsetOutput = compute.length - ((int)Math.pow(2, depth - l - 1) - 1);
            // l = 2, depth = 4
            // 8 4 2 1
            // 2^{4} - 1 - (2^{4 - 2} - 1) = 2^4 - 2^2 = 16 - 4;

            for (int n = 0; n < neuronsInLayer; n += 2) {
                double x1 = compute[offsetInput + n];
                double x2 = compute[offsetInput + n + 1];

                double w1 = weights[offsetInput + n];
                double w2 = weights[offsetInput + n + 1];

                double z = x1 * w1 + x2 * w2;

                compute[offsetOutput + n / 2] = 1.0 / (1.0 + BoundMath.exp(-1 * z));
            }
        }

        return new BasicMLData(new double[] {compute[compute.length - 1]});
    }

    @Override public int getInputCount() {
        return inputCount;
    }

    @Override public int getOutputCount() {
        return 1;
    }

    public int getLayerCount() {
        return depth;
    }

    public Object clone() {
        final TreeNetwork result = (TreeNetwork)ObjectCloner.deepCopy(this);
        return result;
    }

    public int depth() {
        return depth;
    }
}
