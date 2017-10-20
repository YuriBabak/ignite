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

public class PredictedWavWriter implements SequentialOperation {
    private String outputPath;
    private int size;
    private int offset = 0;
    private double[] buff;
    private int outputRate;

    public static double toWavRange(double x) {
        return x * 2 - 1;
    }

    public PredictedWavWriter(String outputPath, int size, int outputRate) {
        this.outputPath = outputPath;
        this.size = size;
        this.outputRate = outputRate;
        buff = new double[size];
    }

    @Override public void init(double[] initial) {
        System.arraycopy(initial, 0, buff, 0, initial.length);
        offset = initial.length;
    }

    @Override public void handle(double[] groundTruth, double[] predicted) {
        System.arraycopy(predicted,0, buff,offset, predicted.length);
        offset+=predicted.length;
    }

    @Override public void finish() {
        WavReader.write(outputPath, Arrays.stream(buff).map(PredictedWavWriter::toWavRange).toArray(), outputRate);
    }
}
