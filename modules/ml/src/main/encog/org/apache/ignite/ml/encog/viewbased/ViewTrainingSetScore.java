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

package org.apache.ignite.ml.encog.viewbased;

import org.encog.mathutil.error.ErrorCalculation;
import org.encog.ml.CalculateScore;
import org.encog.ml.MLMethod;
import org.encog.ml.data.MLDataSet;

public class ViewTrainingSetScore implements CalculateScore {
    private MLDataSet data;

    @Override public double calculateScore(MLMethod method) {
//        final ErrorCalculation errorCalculation = new ErrorCalculation();
//
//        // clear context
//        if( method instanceof MLContext) {
//            ((MLContext)method).clearContext();
//        }
//
//        // calculate error
//        for (final MLDataPair pair : data) {
//            final MLData actual = method.compute(pair.getInput());
//            errorCalculation.updateError(actual.getData(), pair.getIdeal()
//                .getData(),pair.getSignificance());
//        }
//        return errorCalculation.calculate();
        // TODO: WIP
        return 0.0;
    }

    @Override public boolean shouldMinimize() {
        return true;
    }

    @Override public boolean requireSingleThreaded() {
        return false;
    }
}
