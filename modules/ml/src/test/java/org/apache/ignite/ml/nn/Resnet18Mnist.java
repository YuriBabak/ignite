package org.apache.ignite.ml.nn;

import lombok.NoArgsConstructor;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


@NoArgsConstructor
public class Resnet18Mnist extends ZooModel {
    public static final int[] inputShape = new int[] {1, 28, 28};
    public static final int numClasses = 10;

    private long seed;
    private int iterations;

    private WorkspaceMode workspaceMode;
    private ConvolutionLayerIgnite.AlgoMode cudnnAlgoMode;

    public Resnet18Mnist(int iterations, long seed) {
        this(seed, iterations, WorkspaceMode.SEPARATE);
    }

    public Resnet18Mnist(long seed, int iterations, WorkspaceMode workspaceMode) {
        this.seed = seed;
        this.iterations = iterations;
        this.workspaceMode = workspaceMode;
        this.cudnnAlgoMode = workspaceMode == WorkspaceMode.SINGLE ? ConvolutionLayerIgnite.AlgoMode.PREFER_FASTEST
                : ConvolutionLayerIgnite.AlgoMode.NO_WORKSPACE;
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
    }

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        return 0L;
    }

    @Override
    public ZooType zooType() {
        return null;
    }

    @Override
    public Class<? extends Model> modelType() {
        return ComputationGraph.class;
    }

    @Override
    public ComputationGraph init() {
        ComputationGraphConfiguration.GraphBuilder graph = graphBuilder();
        ComputationGraphConfiguration conf = graph.build();
        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        return model;
    }

    private void identityBlock(ComputationGraphConfiguration.GraphBuilder graph, int[] kernelSize, int[] filters,
                               String stage, String block, String input) {
        String convName = "res" + stage + block + "_branch";
        String batchName = "bn" + stage + block + "_branch";
        String activationName = "act" + stage + block + "_branch";
        String shortcutName = "short" + stage + block + "_branch";

        graph.addLayer(convName + "2a",
                new ConvolutionLayerIgnite.Builder(new int[] {1, 1}).nOut(filters[0]).cudnnAlgoMode(cudnnAlgoMode)
                        .build(),
                input)
                .addLayer(batchName + "2a", new BatchNormalization(), convName + "2a")
                .addLayer(activationName + "2a",
                        new ActivationLayerIgnite.Builder().activation(Activation.RELU).build(),
                        batchName + "2a")

                .addLayer(convName + "2b", new ConvolutionLayerIgnite.Builder(kernelSize).nOut(filters[1])
                                .cudnnAlgoMode(cudnnAlgoMode).convolutionMode(ConvolutionMode.Same).build(),
                        activationName + "2a")
                .addLayer(batchName + "2b", new BatchNormalization(), convName + "2b")
                .addLayer(activationName + "2b",
                        new ActivationLayerIgnite.Builder().activation(Activation.RELU).build(),
                        batchName + "2b")

                .addLayer(convName + "2c",
                        new ConvolutionLayerIgnite.Builder(new int[] {1, 1}).nOut(filters[2])
                                .cudnnAlgoMode(cudnnAlgoMode).build(),
                        activationName + "2b")
                .addLayer(batchName + "2c", new BatchNormalization(), convName + "2c")

                .addVertex(shortcutName, new ElementWiseAddVertexIgnite(), batchName + "2c",
                        input)
                .addLayer(convName, new ActivationLayerIgnite.Builder().activation(Activation.RELU).build(),
                        shortcutName);
    }

    private void convBlock(ComputationGraphConfiguration.GraphBuilder graph, int[] kernelSize, int[] filters,
                           String stage, String block, String input) {
        convBlock(graph, kernelSize, filters, stage, block, new int[] {2, 2}, input);
    }

    private void convBlock(ComputationGraphConfiguration.GraphBuilder graph, int[] kernelSize, int[] filters,
                           String stage, String block, int[] stride, String input) {
        String convName = "res" + stage + block + "_branch";
        String batchName = "bn" + stage + block + "_branch";
        String activationName = "act" + stage + block + "_branch";
        String shortcutName = "short" + stage + block + "_branch";

        graph.addLayer(convName + "2a", new ConvolutionLayerIgnite.Builder(new int[] {1, 1}, stride).nOut(filters[0]).build(),
                input)
                .addLayer(batchName + "2a", new BatchNormalization(), convName + "2a")
                .addLayer(activationName + "2a",
                        new ActivationLayerIgnite.Builder().activation(Activation.RELU).build(),
                        batchName + "2a")

                .addLayer(convName + "2b",
                        new ConvolutionLayerIgnite.Builder(kernelSize).nOut(filters[1])
                                .convolutionMode(ConvolutionMode.Same).build(),
                        activationName + "2a")
                .addLayer(batchName + "2b", new BatchNormalization(), convName + "2b")
                .addLayer(activationName + "2b",
                        new ActivationLayerIgnite.Builder().activation(Activation.RELU).build(),
                        batchName + "2b")

                .addLayer(convName + "2c",
                        new ConvolutionLayerIgnite.Builder(new int[] {1, 1}).nOut(filters[2]).build(),
                        activationName + "2b")
                .addLayer(batchName + "2c", new BatchNormalization(), convName + "2c")

                // shortcut
                .addLayer(convName + "1",
                        new ConvolutionLayerIgnite.Builder(new int[] {1, 1}, stride).nOut(filters[2]).build(),
                        input)
                .addLayer(batchName + "1", new BatchNormalization(), convName + "1")


                .addVertex(shortcutName, new ElementWiseAddVertexIgnite(), batchName + "2c",
                        batchName + "1")
                .addLayer(convName, new ActivationLayerIgnite.Builder().activation(Activation.RELU).build(),
                        shortcutName);
    }

    public ComputationGraphConfiguration.GraphBuilder graphBuilder() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                .iterations(iterations).activation(Activation.IDENTITY)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new SGDUpdater(0.1)).weightInit(WeightInit.DISTRIBUTION)
//                .updater(new RmsProp(0.1, 0.96, 0.001)).weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.5)).regularization(true).l1(1e-7).l2(5e-5)
                .convolutionMode(ConvolutionMode.Truncate).graphBuilder();


        graph.addInputs("input").setInputTypes(InputType.convolutionalFlat(inputShape[2], inputShape[1], inputShape[0]))
                // stem
                .addLayer("stem-cnn1",
                        new ConvolutionLayerIgnite.Builder(new int[] {7, 7}, new int[] {2, 2}).nOut(64)
                                .build(),
                        "input")
                .addLayer("stem-batch1", new BatchNormalization(), "stem-cnn1")
                .addLayer("stem-act1", new ActivationLayerIgnite.Builder().activation(Activation.RELU).build(),
                        "stem-batch1")
                .addLayer("stem-maxpool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX,
                        new int[] {3, 3}, new int[] {2, 2}).build(), "stem-act1");

        convBlock(graph, new int[] {3, 3}, new int[] {64, 64, 256}, "2", "a", new int[] {2, 2}, "stem-maxpool1");
        identityBlock(graph, new int[] {3, 3}, new int[] {64, 64, 256}, "2", "b", "res2a_branch");

        graph.addLayer("avgpool",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {3, 3}).build(),
                "res2b_branch")
                // TODO add flatten/reshape layer here
                .addLayer("output",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nOut(numClasses).activation(Activation.SOFTMAX).build(),
                        "avgpool")
                .setOutputs("output").backprop(true).pretrain(false);

        return graph;
    }

    @Override
    public ModelMetaData metaData() {
        return new ModelMetaData(new int[][] {inputShape}, 1, ZooType.CNN);
    }

    @Override
    public void setInputShape(int[][] inputShape) {
        throw new RuntimeException("Resnet18Mnist has input shape fixed.");
    }


    public static void main(String[] args) throws Exception {
        System.setProperty(org.slf4j.impl.SimpleLogger.DEFAULT_LOG_LEVEL_KEY, "INFO");
        Logger log = LoggerFactory.getLogger(Resnet18Mnist.class);

        final int trainBatchSize = 128;
        final int testBatchSize = 128;
        final int epochsNum = 1;

        final int iterations = 20;
        final int seed = 123;


        DataSetIterator mnistTrain = new MnistDataSetIterator(trainBatchSize, true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(testBatchSize, false, seed);

        Resnet18Mnist ml = new Resnet18Mnist(iterations, seed);
        ComputationGraph model = ml.init();

        model.setListeners(new ScoreIterationListener(1));
        for (int i = 0; i != epochsNum; i++) {
            model.fit(mnistTrain);

            Evaluation eval = new Evaluation(Resnet18Mnist.numClasses);
            while (mnistTest.hasNext()) {
                DataSet ds = mnistTest.next();
                INDArray output = model.outputSingle(ds.getFeatureMatrix());
                eval.eval(ds.getLabels(), output);
            }
            log.info(eval.stats());

            mnistTest.reset();
        }
    }
}
