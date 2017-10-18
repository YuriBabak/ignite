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

package org.apache.ignite.ml.encog.util;

import java.util.Arrays;
import org.apache.ignite.ml.encog.wav.WavReader;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

public class GeneratedWavWriter implements SequentialOperation {
    private final int size;
    private final int outputRate;
    private final double[] buff;
    private MLRegression regression;
    private String outputPath;
    private int offset;
    private int histDepth;

    public GeneratedWavWriter(MLRegression regression, String outputPath, int size, int outputRate) {
        this.regression = regression;
        this.outputPath = outputPath;
        this.size = size;
        this.outputRate = outputRate;
        this.buff = new double[size];
    }

    @Override public void init(double[] initial) {
        histDepth = initial.length;
        System.arraycopy(initial, 0, buff, 0, initial.length);
        MLData gen = regression.compute(new BasicMLData(initial));
        offset = initial.length;
        buff[offset] = gen.getData()[0];
        offset++;
    }

    @Override public void handle(double[] groundTruth, double[] predicted) {
        double[] input = new double[histDepth];
        System.arraycopy(buff, offset - histDepth, input, 0, histDepth);
        buff[offset] = regression.compute(new BasicMLData(input)).getData(0);
        offset++;
    }

    @Override public void finish() {
        WavReader.write(outputPath, Arrays.stream(buff).map(PredictedWavWriter::toWavRange).toArray(), outputRate);
    }
}
