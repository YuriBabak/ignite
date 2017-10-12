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

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Locale;
import java.util.UUID;

/**
 * TODO: add description.
 */
public class WavTracer implements SequentialOperation {
    /** */
    private static final String delim = ",";
    /** */
    private final File tmp = createTmpFile();
    /** */
    private final Path path = Paths.get(tmp.toURI());

    private int curSample;
    private int sampleStep;
    private int maxSamples;

    public WavTracer(int sampleStep, int maxSamples) {
        this.sampleStep = sampleStep;
        this.maxSamples = maxSamples;
    }

    @Override public void init(double[] initial) {
        writeHeader();
    }

    @Override public void handle(double[] groundTruth, double[] predicted) {
        assert groundTruth.length == predicted.length;

        if (curSample % sampleStep == 0 && (curSample / sampleStep < maxSamples)) {

            for (int i = 0; i < groundTruth.length; i++)
                try {
                    writeResults(formatResults(groundTruth[i], predicted[i]));
                }
                catch (IOException e) {
                    throw new RuntimeException("Failed to write ");
                }
        }

        curSample++;
    }

    @Override public void finish() {
        //NO-OP.
    }

    /** {@inheritDoc} */
    @Override public String toString() {
        return "WavTracer{" + "tmp=" + tmp.getAbsolutePath() + '}';
    }

    /** */
    private File createTmpFile() {
        final String prefix = UUID.randomUUID().toString(), suffix = ".csv";

        try {
            return File.createTempFile(prefix, suffix);
        }
        catch (IOException e) {
            throw new RuntimeException("Failed to create file [" + prefix + suffix + "].");
        }
    }

    /** */
    private String formatResults(double exp, double actual) {
        assert !formatDouble(1000_000_001.1).contains(delim) : "Formatted results contain [" + delim + "].";

        return "" +
            formatDouble(exp) +
            delim +
            formatDouble(actual) +
            delim +
            formatDouble(Math.abs(exp - actual));
    }

    /** */
    private String formatDouble(double val) {
        return String.format(Locale.US, "%f", val);
    }

    /** */
    private void writeResults(String res) throws IOException {
        final String unixLineSeparator = "\n";

        try (final PrintWriter writer = new PrintWriter(Files.newBufferedWriter(path,
            StandardOpenOption.APPEND, StandardOpenOption.CREATE))) {
            writer.write(res + unixLineSeparator);
        }
    }

    /** */
    private void append(String res) {
        if (res == null)
            throw new IllegalArgumentException("Prediction tracing data is null.");

        try {
            writeResults(res);
        }
        catch (IOException e) {
            throw new RuntimeException("Failed to write to [" + this + "].");
        }
    }

    /** */
    private void writeHeader() {
        append("index" + delim + "expected" + delim + "actual" + delim + "diff");
    }
}
