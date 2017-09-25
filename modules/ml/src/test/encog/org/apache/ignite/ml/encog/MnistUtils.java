package org.apache.ignite.ml.encog;/*
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

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Random;

public class MnistUtils {
    public static class Pair<F, S> {
        F fst;
        S snd;

        public Pair(F fst, S snd) {
            this.fst = fst;
            this.snd = snd;
        }

        public F getFst() {
            return fst;
        }

        public S getSnd() {
            return snd;
        }
    }

    public static Pair<double[][], double[][]> mnist(String imgPath, String labelsPath, Random rnd, int cnt) throws IOException {
        FileInputStream isImages = new FileInputStream(imgPath);
        FileInputStream isLabels = new FileInputStream(labelsPath);

        int magic = read4Bytes(isImages); // Skip magic number.
        int numOfImages = read4Bytes(isImages);
        int imgHeight = read4Bytes(isImages);
        int imgWidth = read4Bytes(isImages);

        read4Bytes(isLabels); // Skip magic number.
        read4Bytes(isLabels); // Skip number of labels.

        int numOfPixels = imgHeight * imgWidth;

        numOfImages /= 60;

        System.out.println("Magic: " + magic);
        System.out.println("Num of images: " + numOfImages);
        System.out.println("Num of pixels: " + numOfPixels);

        Pair<double[][], double[][]> res = new Pair<>(new double[numOfImages][numOfPixels], new double[numOfImages][10]);

        for (int imgNum = 0; imgNum < numOfImages; imgNum++) {
            double[] vec = new double[numOfPixels];

            if (imgNum % 1000 == 0) {
                System.out.println("Read " + imgNum + " images");
            }

            for (int p = 0; p < numOfPixels; p++) {
                int c = 128 - isImages.read();
                vec[p] = (double)c / 128;
            }

            res.getFst()[imgNum] = vec;
            res.getSnd()[imgNum][isLabels.read()] = 1.0;
        }

//        Collections.shuffle(lst, rnd);

        isImages.close();
        isLabels.close();

        return res;
    }

    private static int read4Bytes(FileInputStream is) throws IOException {
        return (is.read() << 24) | (is.read() << 16) | (is.read() << 8) | (is.read());
    }
}