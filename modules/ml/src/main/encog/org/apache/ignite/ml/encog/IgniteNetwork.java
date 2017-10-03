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

import java.util.HashMap;
import java.util.Map;
import org.encog.neural.networks.BasicNetwork;

/**
 * TODO: add description.
 */
public class IgniteNetwork extends BasicNetwork {

    private Map<LockKey, Double> learningMask = new HashMap<>();

    /**
     * @param encoded Encoded.
     * @param mask Mask.
     */
    public void encodeToArray(double[] encoded, Map<LockKey, Double> mask) {
        super.encodeToArray(encoded);

        learningMask = mask;
    }

    /**
     * @param learningMask Learning mask.
     */
    public void setLearningMask(Map<LockKey, Double> learningMask) {
        this.learningMask = learningMask;
    }

    /**
     *
     */
    public Map<LockKey, Double> getLearningMask() {
        return learningMask;
    }

    /** {@inheritDoc} */
    @Override public void setWeight(int fromLayer, int fromNeuron, int toNeuron, double val) {
        double lockVal = learningMask.getOrDefault(new LockKey(fromLayer, fromNeuron, toNeuron), 1d);

        super.setWeight(fromLayer, fromNeuron, toNeuron, val * lockVal);
    }
}
