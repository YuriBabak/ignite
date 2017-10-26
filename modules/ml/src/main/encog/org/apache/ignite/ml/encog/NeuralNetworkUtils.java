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

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;

/**
 * Some NN-related utils.
 */
public class NeuralNetworkUtils {
    /*
     * Return coordinates of neuron in format (layer, neuronInLayer).
     */
    public static int[] toXY(BasicNetwork n, int neuronNumber, boolean isInner) {
        int[] res = new int[2];

        for (int i = isInner ? 1 : 0; i < n.getLayerCount(); i++) {
            if (neuronNumber - n.getLayerNeuronCount(i) < 0) {
                res[0] = i;
                break;
            }
            else
                neuronNumber -= n.getLayerNeuronCount(i);
        }

        res[1] = neuronNumber;

        return res;
    }

    public static int[] toXY(TreeNetwork n, int neuronNumber, boolean isInner) {
        int[] res = new int[2];

        for (int i = isInner ? 1 : 0; i < n.getLayerCount(); i++) {
            if (neuronNumber - n.getLayerNeuronCount(i) < 0) {
                res[0] = i;
                break;
            }
            else
                neuronNumber -= n.getLayerNeuronCount(i);
        }

        res[1] = neuronNumber;

        return res;
    }

    public static String printBinaryNetwork(BasicNetwork nn) {
        StringBuilder sb = new StringBuilder();
        for (int l = 0; l < nn.getLayerCount() - 1; l++) {
            sb.append("\n");
            for (int n = 0; n < nn.getLayerNeuronCount(l); n += 2) {
                sb.append("[")
                    .append(nn.getWeight(l, n, n / 2))
                    .append(",")
                    .append(nn.getWeight(l, n + 1, n / 2))
                    .append("]");
            }
        }
        return sb.toString();
    }

    public static int totalNeuronsCount(BasicNetwork n) {
        int res = 0;

        for (int i = 0; i < n.getLayerCount(); i++)
            res += n.getLayerNeuronCount(i);

        return res;
    }

    public static int innerNeuronsCount(BasicNetwork n) {
        return totalNeuronsCount(n) - n.getLayerNeuronCount(0) - n.getLayerNeuronCount(n.getLayerCount() - 1);
    }

    public static int innerNeuronsCount(TreeNetwork n) {
        return ((int)Math.pow(2, n.depth) - 1) - n.getLayerNeuronCount(0) - n.getLayerNeuronCount(n.getLayerCount() - 1);
    }

    protected static IgniteNetwork buildTreeLikeNetComplex(int leavesCountLog, int lookForwardCnt) {
        IgniteNetwork res = new IgniteNetwork();

        int lastTreeLike = 0;
        for (int i = leavesCountLog; i >=0 && Math.pow(2, i) >= (lookForwardCnt / 2); i--) {
            res.addLayer(new BasicLayer(i == 0 ? null : new ActivationSigmoid(), false, (int)Math.pow(2, i)));
            lastTreeLike = leavesCountLog - i;
        }

        res.addLayer(new BasicLayer(new ActivationSigmoid(), false, lookForwardCnt));

        res.getStructure().finalizeStructure();

        for (int i = 0; i < lastTreeLike; i++) {
            for (int n = 0; n < res.getLayerNeuronCount(i); n += 2) {
                res.dropOutputsFrom(i, n);
                res.dropOutputsFrom(i, n + 1);

                res.enableConnection(i, n, n / 2, true);
                res.enableConnection(i, n + 1, n / 2, true);
            }
        }

        res.reset();

        return res;
    }
}
