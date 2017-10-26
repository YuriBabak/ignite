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

import java.io.File;
import java.io.IOException;
import java.util.Random;
import org.apache.ignite.ml.encog.util.PredictedWavWriter;
import org.apache.ignite.ml.encog.wav.WavReader;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.persist.PersistorRegistry;
import org.junit.Test;

public class GenerateWav {
    private static final String WAV_FOLDER = "/home/enny/";

    @Test
    public void test() throws IOException, ClassNotFoundException {
        String nnName = "model1508606358976.nn";
        int generateSamples = 1_000_000;
        int rate = 22100;

        PersistorRegistry.getInstance().add(new PersistIgniteNetwork());
        BasicNetwork net = (BasicNetwork)EncogDirectoryPersistence.loadObject(new File(WAV_FOLDER + nnName));
        System.out.println("Network input size: " + net.getInputCount() + ".");
        String outPath = WAV_FOLDER + "gen" + System.currentTimeMillis() + ".wav";

        double[] buff = new double[generateSamples];
        double[] input = getInput(net.getInputCount());

        for (int i = 0; i < generateSamples; i++) {
            double v = net.compute(new BasicMLData(input)).getData()[0];
            buff[i] = PredictedWavWriter.toWavRange(v);

            System.arraycopy(input, 1, input, 0, input.length - 1);
            input[input.length - 1] = v;
        }

        WavReader.write(outPath, buff, rate);
    }

    private double[] getInput(int inputSize) {
        double[] res = new double[inputSize];

        for (int i = 0; i < inputSize; i++)
            res[i] = new Random().nextDouble();
            return res;
    }
}