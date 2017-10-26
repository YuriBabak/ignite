package org.apache.ignite.ml.nn.layers;

import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.internal.util.typedef.T3;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.apache.ignite.ml.nn.gradient.DefaultGradient;
import org.apache.ignite.ml.nn.gradient.Gradient;
import org.apache.ignite.ml.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.apache.ignite.ml.nn.util.LossCalculation;
import org.apache.ignite.ml.nn.util.LossFunction;

import org.apache.ignite.ml.math.Matrix;
import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.nn.util.Algorithms;



public abstract class BaseOutputLayer<LayerConfT extends org.apache.ignite.ml.nn.conf.layers.BaseOutputLayer>
        extends BaseLayer<LayerConfT> {

    protected INDArray labels;

    private double fullNetworkL1;
    private double fullNetworkL2;

    public BaseOutputLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public double computeScore( double fullNetworkL1, double fullNetworkL2) {
        if( input == null || labels == null )
            throw new IllegalStateException("Cannot calculate score without input and labels");
        this.fullNetworkL1 = fullNetworkL1;
        this.fullNetworkL2 = fullNetworkL2;
        INDArray preOut = preOutput(true);
        LossFunction lf = ((org.apache.ignite.ml.nn.conf.layers.BaseOutputLayer)conf.getLayer()).getLossFunction();
        if ( (lf == LossFunction.NEGATIVELOGLIKELIHOOD || lf == LossFunction.MCXENT) && layerConf().getActivationFunction().equals("softmax")) {
            setScore(null,preOut);
        }
        return score;
    }

    private void setScore(INDArray z, INDArray preOut ) {
        score = LossCalculation.builder()
                .l1(fullNetworkL1).l2(fullNetworkL2)
                .labels(getLabels2d()).z(z)
                .preOut(preOut).activationFn(conf().getLayer().getActivationFunction())
                .lossFunction(layerConf().getLossFunction())
                .miniBatch(conf.isMiniBatch()).miniBatchSize(getInputMiniBatchSize())
                .useRegularization(conf.isUseRegularization())
                .build().score();
    }

    @Override
    public IgniteBiTuple<Gradient, Double> gradientAndScore() {
        throw new RuntimeException("Method gradientAndScore is not implemented.");
    }

    @Override
    public IgniteBiTuple<Gradient,INDArray> backpropGradient(INDArray epsilon) {
        T3<Gradient,INDArray,INDArray> triple = getGradientsAndDelta(preOutput(true));
        INDArray delta = triple.get2();
        Matrix deltaT = Algorithms.toIgnite(delta);

        Matrix epsilonNextT = Algorithms.toIgnite(params.get(DefaultParamInitializer.WEIGHT_KEY));
        epsilonNextT = epsilonNextT.times(deltaT.transpose()).transpose();
        INDArray epsilonNext = params.get(DefaultParamInitializer.WEIGHT_KEY).mmul(delta.transpose()).transpose();

        return new IgniteBiTuple<>(triple.get1(), Algorithms.toNd4j(epsilonNextT));
    }

    @Override
    public Gradient gradient() {
        throw new RuntimeException("Method gradient is not implemented.");
    }

    private T3<Gradient,INDArray,INDArray> getGradientsAndDelta(INDArray preOut) {
        String afn = conf().getLayer().getActivationFunction();
        Matrix output;
        if ("softmax".equals(afn)) {
            output = Algorithms.toIgnite(preOut);
            output = output.map((x) -> Math.exp(x));

            for (int i = 0; i != output.rowSize(); ++i) {
                Vector row = output.viewRow(i);
                row = row.divide(row.sum());
            }
        } else {
            throw new RuntimeException("Unsupported activation function.");
        }

        // TODO: just subtraction of matrices.
        Matrix outSubLabels = output.plus(Algorithms.toIgnite(getLabels2d()).times(-1));
        Gradient gradient = new DefaultGradient();

        INDArray weightGradView = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY);
        INDArray biasGradView = gradientViews.get(DefaultParamInitializer.BIAS_KEY);

        gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY,weightGradView);
        gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY,biasGradView);

        T3<Gradient,INDArray,INDArray> triple;
        switch (layerConf().getLossFunction()) {
            case NEGATIVELOGLIKELIHOOD:
            case MCXENT:
                Matrix weightGradViewT = Algorithms.toIgnite(input).transpose().times(outSubLabels);
                weightGradView.assign(Algorithms.toNd4j(weightGradViewT));

                biasGradView.assign(Algorithms.toNd4j(Algorithms.sumRows(outSubLabels)));
                triple = new T3<>(gradient, Algorithms.toNd4j(outSubLabels), Algorithms.toNd4j(output));
                break;

            default:
                throw new IllegalStateException("Unsupported loss function.");
        }

        return triple;
    }

    public  void setLabels(INDArray labels) {
        this.labels = labels;
    }

    protected INDArray getLabels2d(){
        assert(labels.rank() == 2);
        return labels;
    }
}
