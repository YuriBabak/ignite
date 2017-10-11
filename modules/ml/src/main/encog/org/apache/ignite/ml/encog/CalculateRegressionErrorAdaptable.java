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

import org.encog.mathutil.error.ErrorCalculation;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.BasicNetwork;

public class CalculateRegressionErrorAdaptable {
    public static double calculateError(final MLRegression method,
        final MLDataSet data) {
        final ErrorCalculation errorCalculation = new ErrorCalculation();

        BasicNetwork nw = (BasicNetwork)method;

        // calculate error
        int ratio = data.get(0).getInput().size() / nw.getLayerNeuronCount(0);

        for (final MLDataPair pair : data) {

            final MLData actual = method.compute(adaptData(pair.getInput(), ratio));
            errorCalculation.updateError(actual.getData(), pair.getIdeal()
                .getData(),pair.getSignificance());
        }
        return errorCalculation.calculate();
    }

    private static MLData adaptData(MLData original, int ratio) {
        if (ratio == 1)
            return original;
        double[] avgBuff = new double[original.size() / ratio];

        for (int i = 0; i <= original.size() - ratio; i += ratio) {
            double avg = 0.0;
            for (int j = 0; j < ratio; j++)
                avg += original.getData(i + j);
            avgBuff[i / ratio] = avg / ratio;
        }

        return new BasicMLData(avgBuff);
    }
}
