package org.apache.ignite.examples.ml;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;


import org.apache.ignite.ml.nn.api.OptimizationAlgorithm;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.apache.ignite.ml.nn.conf.Updater;
import org.apache.ignite.ml.nn.conf.ComputationGraphConfiguration;
import org.apache.ignite.ml.nn.conf.inputs.InputType;
import org.apache.ignite.ml.nn.conf.layers.*;
import org.apache.ignite.ml.nn.graph.ComputationGraph;
import org.apache.ignite.ml.nn.conf.graph.ElementWiseVertex;
import org.apache.ignite.ml.nn.weights.WeightInit;
import org.apache.ignite.ml.nn.optimize.listeners.ScoreIterationListener;
import org.apache.ignite.ml.nn.util.LossFunction;
//import org.deeplearning4j.nn.api.OptimizationAlgorithm;
//import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
//import org.deeplearning4j.nn.conf.Updater;
//import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
//import org.deeplearning4j.nn.conf.inputs.InputType;
//import org.deeplearning4j.nn.conf.layers.*;
//import org.deeplearning4j.nn.graph.ComputationGraph;
//import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
//import org.deeplearning4j.nn.weights.WeightInit;
//import org.deeplearning4j.optimize.listeners.ScoreIterationListener;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class Resnet18Mnist_simplified {
    private static final Logger log = LoggerFactory.getLogger(Resnet18Mnist_simplified.class);

    public static void main(String[] args) throws Exception {
        int inputWidth = 28;
        int inputHeight = 28;
        int inputChannels = 1;
        int outputsNum = 10;
        int trainBatchSize = 64;
        int testBatchSize = 32;
        int epochsNum = 10;

        int seed = 123;
        int iterations = 1;
        double learningRate = 0.01;
        double l2_regular = 0.0005;

        log.info("Loading the data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(trainBatchSize,true,12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(testBatchSize,false,12345);

        log.info("Building the model....");


        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(l2_regular)
                .learningRate(learningRate)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .graphBuilder();

        graph.addInputs("Input")
                .setInputTypes(InputType.convolutional(inputHeight,inputWidth, inputChannels))
                .addLayer("Conv1", new ConvolutionLayer.Builder(3, 3)
                        .nIn(inputChannels)
                        .stride(1, 1)
                        .nOut(32)
                        .activation("relu")
                        .build(), "Input")
                .addLayer("Conv2", new ConvolutionLayer.Builder(1, 1)
                        .stride(1, 1)
                        .nIn(32)
                        .nOut(64)
                        .activation("relu")
                        .build(), "Conv1")
                .addLayer("Conv3", new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .padding(1, 1)
                        .nIn(64)
                        .nOut(32)
                        .activation("relu")
                        .build(), "Conv2")
                .addLayer("Conv4", new ConvolutionLayer.Builder(1, 1)
                        .stride(1, 1)
                        .nIn(32)
                        .nOut(64)
                        .activation("relu")
                        .build(), "Conv3")
                .addLayer("Activation1", new ActivationLayer.Builder()
                        .nOut(64)
                        .activation("relu")
                        .build(), "Conv4")
                .addLayer("Activation2", new ActivationLayer.Builder()
                        .nOut(64)
                        .activation("relu")
                        .build(), "Conv2")
                .addVertex("Add", new ElementWiseVertex(ElementWiseVertex.Op.Add),
                        "Activation1", "Activation2")
                .addLayer("Output", new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(43264)
                        .nOut(outputsNum)
                        .activation("softmax")
                        .build(), "Add")
                .setOutputs("Output");

        ComputationGraphConfiguration conf = graph.build();
        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        log.info("Training....");
        model.setListeners(new ScoreIterationListener(1));
        for (int i = 0; i < epochsNum; i++) {
            model.fit(mnistTrain);

            log.info("Epoch {} completed", i);

            log.info("Evaluation....");
            Evaluation eval = new Evaluation(outputsNum);
            DataSet ds = mnistTest.next();

            INDArray[] output = model.output(ds.getFeatureMatrix());
            eval.eval(ds.getLabels(), output[0]);

            log.info(eval.stats());
            mnistTest.reset();
        }
        log.info("Done!");
    }
}