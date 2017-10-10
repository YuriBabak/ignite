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

package org.apache.ignite.ml.encog.wav;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import org.apache.ignite.IgniteException;

/**
 * Wav reader
 */
public class WavReader {
    /**
     * Read wav file to list of batch of wav frames.
     *
     * @param path Path.
     * @param numOfFramesInBatch Number of frames in batch.
     */
    public static List<double[]> read(String path, int numOfFramesInBatch){
        List<double[]> batchs = new ArrayList<>();

        try {
            WavFile wavFile = WavFile.openWavFile(new File(path));
            System.out.println("");wavFile.getValidBits();

            wavFile.display();

            int numChannels = wavFile.getNumChannels();
            double[] buf = new double[numOfFramesInBatch * numChannels];

            int framesRead;

            do {
                framesRead = wavFile.readFrames(buf, numOfFramesInBatch);

                batchs.add(buf.clone());
            } while (framesRead != 0);

            wavFile.close();
        }
        catch (IOException | WavFileException e) {
            throw new IgniteException("Failed to read file: " + path, e);
        }

        return batchs;
    }

    // TODO: at the moment we support only single channeled files.
    public static double[] readAsSingleChannel(String path){
        double[] arr = null;
        try {
            WavFile wavFile = WavFile.openWavFile(new File(path));
            int frames = (int)(wavFile.getNumFrames() / wavFile.getNumChannels());
            arr = new double[frames];
            System.out.println("");
            wavFile.getValidBits();
            wavFile.display();

            int framesRead;

            do {
                framesRead = wavFile.readFrames(arr, frames);
            } while (framesRead != 0);

            wavFile.close();
        }
        catch (IOException | WavFileException e) {
            throw new IgniteException("Failed to read file: " + path, e);
        }

        return arr;
    }

    public static void write(String path, double[] res, int rate){
        List<double[]> batchs = new ArrayList<>();

        try {
            WavFile wavFile = WavFile.newWavFile(new File(path), 1, res.length, 16, rate / 2);
            wavFile.writeFrames(res, 0, res.length);

            wavFile.close();
        }
        catch (IOException | WavFileException e) {
            throw new IgniteException("Failed to read file: " + path, e);
        }
    }

}
