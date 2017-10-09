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

import org.encog.ml.data.MLData;
import org.encog.util.kmeans.Centroid;

public class MLDataView implements MLData {
    private final int offset;
    private final int length;
    double[] underlying;

    public MLDataView(double[] underlying, int offset, int length) {
        this.underlying = underlying;
        this.offset = offset;
        this.length = length;
    }

    @Override public void add(int index, double value) {
        throw new UnsupportedOperationException();
    }

    @Override public void clear() {
        throw new UnsupportedOperationException();
    }

    @Override public MLData clone() {
        throw new UnsupportedOperationException();
    }

    // TODO: this method is used in TrainingSet error class, should rewrite this class
    // to get rid of copying
    @Override public double[] getData() {
        throw new UnsupportedOperationException();
    }

    @Override public double getData(int index) {
        return underlying[offset + index];
    }

    @Override public void setData(double[] data) {
        throw new UnsupportedOperationException();
    }

    @Override public void setData(int index, double d) {
        throw new UnsupportedOperationException();
    }

    @Override public int size() {
        return length;
    }

    @Override public Centroid<MLData> createCentroid() {
        return null;
    }

    public double[] getUnderlying() {
        return underlying;
    }
}
