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

import java.io.Serializable;
import org.apache.ignite.Ignite;
import org.apache.ignite.ml.encog.caches.TrainingContext;
import org.encog.ml.ea.opp.EvolutionaryOperator;
import org.encog.ml.ea.train.EvolutionaryAlgorithm;

public abstract class IgniteEvolutionaryOperator implements EvolutionaryOperator, Serializable {
    private TrainingContext ctx;
    private Ignite ignite;
    private double prob;
    private String operatorId;

    private EvolutionaryAlgorithm owner;

    public IgniteEvolutionaryOperator(double prob, String operatorId) {
        this.prob = prob;
        this.operatorId = operatorId;
    }

    @Override public void init(EvolutionaryAlgorithm theOwner) {
        this.owner = theOwner;
    }

    public void setContext(TrainingContext ctx) {
        this.ctx = ctx;
    }

    public TrainingContext context() {
        return ctx;
    }

    public Ignite ignite() {
        return ignite;
    }

    public void setIgnite(Ignite ignite) {
        this.ignite = ignite;
    }

    public double probability() {
        return prob;
    }

    public String operatorId() {
        return operatorId;
    }
}
