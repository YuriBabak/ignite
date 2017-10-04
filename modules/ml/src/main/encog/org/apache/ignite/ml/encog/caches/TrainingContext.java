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

package org.apache.ignite.ml.encog.caches;

import java.io.Serializable;
import java.util.Map;
import org.apache.ignite.ml.encog.GATrainerInput;
import org.apache.ignite.ml.encog.SubPopulationStatistics;
import org.encog.ml.MLMethod;

// TODO: maybe we should store in cache only input and all other, non-constant data should be send each time.
public class TrainingContext<S, U extends Serializable> implements Serializable {
    private int currentIteration;
    private Map<Integer, SubPopulationStatistics> subPopulationStatistics;

    public TrainingContext() {

    }

    public TrainingContext(GATrainerInput<? extends MLMethod, S, U> input) {
    }
}
