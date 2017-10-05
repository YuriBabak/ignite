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

import org.encog.ml.MLMethod;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.training.TrainingSetScore;

public class DebugTrainingSetScore extends TrainingSetScore {
    /**
     * Construct a training set score calculation.
     *
     * @param training The training data to use.
     */
    public DebugTrainingSetScore(MLDataSet training) {
        super(training);
    }

    @Override public double calculateScore(MLMethod method) {
        double sc = super.calculateScore(method);
        System.out.println("Score: " + sc);
        return sc;
    }
}
