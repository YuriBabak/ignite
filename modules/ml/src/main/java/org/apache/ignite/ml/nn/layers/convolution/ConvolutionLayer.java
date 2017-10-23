package org.apache.ignite.ml.nn.layers.convolution;

import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.math.Matrix;
import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.math.impls.matrix.DenseLocalOnHeapMatrix;
import org.apache.ignite.ml.math.impls.vector.DenseLocalOnHeapVector;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.apache.ignite.ml.nn.gradient.DefaultGradient;
import org.apache.ignite.ml.nn.gradient.Gradient;
import org.apache.ignite.ml.nn.layers.BaseLayer;
import org.apache.ignite.ml.nn.params.ConvolutionParamInitializer;
import org.apache.ignite.ml.nn.util.Algorithms;
import org.apache.ignite.ml.nn.util.Tensor4d;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class ConvolutionLayer extends BaseLayer<org.apache.ignite.ml.nn.conf.layers.ConvolutionLayer> {
    protected static final Logger log = LoggerFactory.getLogger(ConvolutionLayer.class);

    public ConvolutionLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public double calcL2() {
    	if(!conf.isUseRegularization() || conf.getLayer().getL2() <= 0.0 ) return 0.0;

        double l2Norm = getParam(ConvolutionParamInitializer.WEIGHT_KEY).norm2Number().doubleValue();
        return 0.5 * conf.getLayer().getL2() * l2Norm * l2Norm;
    }

    @Override
    public double calcL1() {
    	if(!conf.isUseRegularization() || conf.getLayer().getL1() <= 0.0 ) return 0.0;
        return conf.getLayer().getL1() * getParam(ConvolutionParamInitializer.WEIGHT_KEY).norm1Number().doubleValue();
    }

    @Override
    public Type type() {
        return Type.CONVOLUTIONAL;
    }


    @Override
    public IgniteBiTuple<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        INDArray weights = getParam(ConvolutionParamInitializer.WEIGHT_KEY);
        INDArray biasGradView = gradientViews.get(ConvolutionParamInitializer.BIAS_KEY);
        INDArray weightGradView = gradientViews.get(ConvolutionParamInitializer.WEIGHT_KEY);

        //////////////////////////////
        Tensor4d weightsT = Tensor4d.toIgnite(weights);
        Tensor4d epsilonT = Tensor4d.toIgnite(epsilon);
        Matrix biasGradViewT = Algorithms.toIgnite(biasGradView);
        Tensor4d weightGradViewT = Tensor4d.toIgnite(weightGradView);
        //////////////////////////////

        int miniBatch = input.size(0);
        int inH = input.size(2);
        int inW = input.size(3);

        int outDepth = weightsT.dims[0];
        int inDepth = weightsT.dims[1];
        int kH = weightsT.dims[2];
        int kW = weightsT.dims[3];

        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();
        int[] pad = layerConf().getPadding();
        assert(kernel[0] == kernel[1]);
        assert(strides[0] == strides[1]);
        assert(pad[0] == pad[1]);

        int outH = Algorithms.convOutSize(inH, kernel[0], strides[0], pad[0]);
        int outW = Algorithms.convOutSize(inW, kernel[1], strides[1], pad[1]);


        String afn = conf.getLayer().getActivationFunction();
        Tensor4d deltaT;

        if("identity".equals(afn)){
            deltaT = Tensor4d.toIgnite(epsilon);

        } else {  // relu
            INDArray sigmaPrimeZ = preOutput(true);
            Tensor4d sigmaPrimeZT = Tensor4d.toIgnite(sigmaPrimeZ);

            deltaT = Algorithms.applyTo("step", sigmaPrimeZT);
            deltaT.muli(epsilonT);
        }
        Matrix delta2dT = new DenseLocalOnHeapMatrix(outDepth, miniBatch * outH * outW);

        // TODO: algorithm below is just reshape.
        for (int d = 0; d != outDepth; ++d) {
            Vector row = delta2dT.viewRow(d);

            int step = outH * outW;
            for (int b = 0; b != miniBatch; ++b) {
                Vector stacked = Algorithms.vec(deltaT.data[b][d].transpose());

                row.viewPart(b * step, step).assign(stacked);
            }
        }

        Matrix im2col2dT = Algorithms.batch2col(Tensor4d.toIgnite(input), kernel[0], strides[0], pad[0]);
        Matrix weightGradView2dfT = im2col2dT.times(delta2dT.transpose());

        Matrix w2dT = new DenseLocalOnHeapMatrix(inDepth*kH*kW, outDepth);

        // TODO: algorithm below is just permute + reshape.
        for (int d0 = 0; d0 != outDepth; ++d0) {
            Vector column = w2dT.viewColumn(d0);

            int step = kH * kW;
            for (int d1 = 0; d1 != inDepth; ++d1) {
                Vector stacked = Algorithms.vec(weightsT.data[d0][d1].transpose());

                column.viewPart(d1 * step, step).assign(stacked);
            }
        }

        Matrix epsNext2dT = w2dT.times(delta2dT);

        Tensor4d batch = Algorithms.col2batch(epsNext2dT, new int[]{miniBatch, inDepth, inH, inW}, kernel[0], strides[0], pad[0]);

        Gradient retGradient = new DefaultGradient();
        {
            Vector biasGradTempT = new DenseLocalOnHeapVector(delta2dT.rowSize());
            for (int i = 0; i != delta2dT.columnSize(); ++i) {
                Vector v = delta2dT.viewColumn(i);

                biasGradTempT = biasGradTempT.plus(v);
            }
            biasGradView.assign(Algorithms.toNd4j(biasGradTempT));
            weightGradView.assign(Algorithms.toNd4j(weightGradView2dfT));
        }


        retGradient.setGradientFor(ConvolutionParamInitializer.BIAS_KEY, biasGradView);
        retGradient.setGradientFor(ConvolutionParamInitializer.WEIGHT_KEY, weightGradView, 'c');

        /////////////////////////////
        INDArray epsNext = batch.toNd4j();
        /////////////////////////////

        return new IgniteBiTuple<>(retGradient,epsNext);
    }

    public INDArray preOutput(boolean training) {
        INDArray weights = getParam(ConvolutionParamInitializer.WEIGHT_KEY);
        INDArray bias = getParam(ConvolutionParamInitializer.BIAS_KEY);

        ////////////////////////
        Tensor4d weightsT = Tensor4d.toIgnite(weights);
        Matrix biasT = Algorithms.toIgnite(bias);
        ////////////////////////

        int miniBatch = input.size(0);
        int inH = input.size(2);
        int inW = input.size(3);

        int outDepth = weightsT.dims[0];
        int inDepth = weightsT.dims[1];
        int kH = weightsT.dims[2];
        int kW = weightsT.dims[3];

        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();
        int[] pad = layerConf().getPadding();
        assert(kernel[0] == kernel[1]);
        assert(strides[0] == strides[1]);
        assert(pad[0] == pad[1]);

        int outH = Algorithms.convOutSize(inH, kernel[0], strides[0], pad[0]);
        int outW = Algorithms.convOutSize(inW, kernel[1], strides[1], pad[1]);


        Matrix im2col2dT = Algorithms.batch2col(Tensor4d.toIgnite(input), kernel[0], strides[0], pad[0]);

        // TODO: algorithm below is just permute + reshape.
        Matrix w2dT = new DenseLocalOnHeapMatrix(inDepth*kH*kW, outDepth);
        for (int d0 = 0; d0 != outDepth; ++d0) {
            Vector column = w2dT.viewColumn(d0);

            int step = kH * kW;
            for (int d1 = 0; d1 != inDepth; ++d1) {
                Vector stacked = Algorithms.vec(weightsT.data[d0][d1].transpose());

                column.viewPart(d1 * step, step).assign(stacked);
            }
        }

        Matrix zT = im2col2dT.transpose().times(w2dT);

        // TODO: algorithm below is just row-wise addition with vector.
        for (int i = 0; i != zT.rowSize(); ++i) {
            Vector row = zT.viewRow(i);
            row = row.plus(biasT.viewRow(0));
        }

        Tensor4d batch = new Tensor4d();
        batch.data = new Matrix[miniBatch][outDepth];
        batch.dims = new int[]{ miniBatch, outDepth, outW, outH };
        for (int i = 0; i != miniBatch; ++i) {
            for (int j = 0; j != outDepth; ++j) {
                batch.data[i][j] = new DenseLocalOnHeapMatrix(outH, outW);
            }
        }

        // TODO: algorithm below is just reshape + permute.
        for (int d = 0; d != outDepth; ++d) {
            int step = outH * outW;

            for (int b = 0; b != miniBatch; ++b) {
                Vector v = zT.viewColumn(d).viewPart(b * step, step);

                batch.data[b][d].assign(Algorithms.unvec(v, outW).transpose());
            }
        }

        return batch.toNd4j();
    }

    @Override
    public INDArray activate(boolean training) {
        if(input == null) {
            throw new IllegalArgumentException("No null input allowed");
        }

        Tensor4d z = Tensor4d.toIgnite(preOutput(training));

        String afn = conf.getLayer().getActivationFunction();
        if ("identity".equals(afn)) {
            return z.toNd4j();
        } else if ("relu".equals(afn)) {
            z = Algorithms.applyTo("relu", z);
        }

        return z.toNd4j();
    }

    @Override
    public INDArray params() {
        throw new RuntimeException("Not implemented.");
    }

    @Override
    public void setParams(INDArray params){
        setParams(params,'c');
    }
}
