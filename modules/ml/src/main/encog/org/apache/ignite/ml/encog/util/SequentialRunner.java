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

import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import org.apache.ignite.ml.encog.EncogMethodWrapper;
import org.apache.ignite.ml.encog.wav.WavReader;
import org.encog.ml.data.basic.BasicMLData;

// TODO: Can be generalized not only on WAV files
public class SequentialRunner {
    List<SequentialOperation> ops = new LinkedList<>();

    public void add(SequentialOperation op) {
        ops.add(op);
    }

    public void run(EncogMethodWrapper model, int histDepth, int numOfFramesInBatch, String testWavPath ) throws IOException {
        WavReader.WavInfo info = WavReader.read(testWavPath, numOfFramesInBatch);
        List<double[]> ds = info.batchs();
        int channels = info.file().getNumChannels();

        // Init all operations
        double[] initial = ds.subList(0, histDepth).stream().map(doubles ->
            (Arrays.stream(doubles).sum() / doubles.length + 1) / channels).mapToDouble(d -> d).toArray();

        ops.forEach(op -> op.init(initial));

        int samplesCnt = ds.size();

        for (int i = histDepth; i < samplesCnt - 1; i++){
            BasicMLData dataSetEntry = new BasicMLData(ds.subList(i - histDepth, i).stream().map(doubles ->
                (Arrays.stream(doubles).sum() / doubles.length + 1) / channels).mapToDouble(d -> d).toArray());

            double[] rawLable = ds.get(i + 1);
            double[] lable = {(Arrays.stream(rawLable).sum() / rawLable.length + 1) / channels};
            double[] predict = model.predict(dataSetEntry);

            ops.forEach(operation -> operation.handle(lable, predict));
        }

        ops.forEach(SequentialOperation::finish);
    }
}
