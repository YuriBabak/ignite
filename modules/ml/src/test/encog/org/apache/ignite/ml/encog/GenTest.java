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

import java.io.IOException;
import java.util.Random;
import org.apache.ignite.Ignite;
import org.apache.ignite.configuration.IgniteConfiguration;
import org.apache.ignite.internal.util.IgniteUtils;
import org.apache.ignite.ml.math.functions.IgniteSupplier;
import org.apache.ignite.testframework.junits.IgniteTestResources;
import org.apache.ignite.testframework.junits.common.GridCommonAbstractTest;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.junit.Test;

public class GenTest  extends GridCommonAbstractTest {
    public static final String MNIST_LOCATION = "C:/Users/Yury/Downloads/mnist/";
    private static final int NODE_COUNT = 4;

    /** Grid instance. */
    protected Ignite ignite;

    /**
     * Default constructor.
     */
    public GenTest() {
        super(false);
    }

    /**
     * {@inheritDoc}
     */
    @Override protected void beforeTest() throws Exception {
        ignite = grid(NODE_COUNT);
    }

    /** {@inheritDoc} */
    @Override protected void beforeTestsStarted() throws Exception {
        for (int i = 1; i <= NODE_COUNT; i++)
            startGrid(i);
    }

    /** {@inheritDoc} */
    @Override protected void afterTestsStopped() throws Exception {
        stopAllGrids();
    }

    @Override protected long getTestTimeout() {
        return 6000000;
    }

    @Override protected IgniteConfiguration getConfiguration(String igniteInstanceName,
        IgniteTestResources rsrcs) throws Exception {
        IgniteConfiguration configuration = super.getConfiguration(igniteInstanceName, rsrcs);
        configuration.setMarshaller(null);
        configuration.setIncludeEventTypes();
        configuration.setPeerClassLoadingEnabled(true);
        configuration.setMetricsUpdateFrequency(2000);

        return configuration;
    }

    @Test
    public void test() throws IOException {
        IgniteUtils.setCurrentIgniteName(ignite.configuration().getIgniteInstanceName());

        System.out.println("Reading mnist...");
        MnistUtils.Pair<double[][], double[][]> mnist = MnistUtils.mnist(MNIST_LOCATION + "train-images-idx3-ubyte", MNIST_LOCATION + "train-labels-idx1-ubyte", new Random(), 60_000);
        System.out.println("Done");

        // create training data
        MLDataSet trainingSet = new BasicMLDataSet(mnist.fst, mnist.snd);
        IgniteSupplier<BasicNetwork> fact = () -> {
            BasicNetwork res = new BasicNetwork();
            res.addLayer(new BasicLayer(null,true,2));
            res.addLayer(new BasicLayer(new org.encog.engine.network.activation.ActivationSigmoid(),false,5));
            res.addLayer(new BasicLayer(new org.encog.engine.network.activation.ActivationSigmoid(),false,1));
            res.getStructure().finalizeStructure();

            res.reset();
            return res;
        };

        GaTrainerInput input = new GaTrainerInput(trainingSet, fact);

        EncogMethodWrapper model = new GATrainer(ignite).train(input);

//        model.predict();
    }
}
