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
import org.apache.ignite.ml.encog.GATrainerInput;
import org.apache.ignite.ml.math.functions.IgniteSupplier;
import org.encog.ml.MLEncodable;
import org.encog.ml.MLMethod;
import org.encog.ml.MethodFactory;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.ea.genome.GenomeFactory;
import org.encog.ml.factory.MLMethodFactory;

public class TrainingContext implements Serializable {
    private IgniteSupplier<MLMethod> methodFactory;
    private GenomeFactory factory;
    private int datasetSize;
    private GATrainerInput<? extends MLMethod> input;

    public TrainingContext() {

    }

    public TrainingContext(GenomeFactory genomeFactory, GATrainerInput<? extends MLMethod> input) {
        this.factory = genomeFactory;
        this.input = input;
        this.datasetSize = datasetSize;
    }

    public GenomeFactory genomeFactory() {
        return factory;
//        return null;
    }

    public MethodFactory getMlMethodFactory() {
//        return () -> methodFactory.get();
        return () -> input.methodFactory().get();
//        return null;
    }

    public int datasetSize() {
        return datasetSize;
    }

    public GATrainerInput<? extends MLMethod> input() {
        return input;
    }
}
