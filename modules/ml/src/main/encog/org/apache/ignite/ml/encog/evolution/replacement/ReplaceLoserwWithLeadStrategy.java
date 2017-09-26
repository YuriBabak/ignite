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

package org.apache.ignite.ml.encog.evolution.replacement;

import java.util.ArrayList;
import java.util.List;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.population.Population;

public class ReplaceLoserwWithLeadStrategy implements UpdateStrategy {
    private double replacePercentage;

    public ReplaceLoserwWithLeadStrategy(double replacePercentage) {
        this.replacePercentage = replacePercentage;
    }

    @Override public List<Genome> getNewGenomes(Population population, Genome best) {
        int size = population.getPopulationSize();
        int cntToReplace = (int)(size * replacePercentage);

        int i = 0;
        List<Genome> res = new ArrayList<>(size);

        for (Genome genome : population.getSpecies().get(0).getMembers()) {
            if (i < cntToReplace)
                res.add(best);
            else
                res.add(genome);
            i++;
        }

        return res;
    }
}
