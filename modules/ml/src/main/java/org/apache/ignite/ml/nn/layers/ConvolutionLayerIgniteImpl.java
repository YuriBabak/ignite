package org.apache.ignite.ml.nn.layers;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.convolution.ConvolutionLayer;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;


public class ConvolutionLayerIgniteImpl extends ConvolutionLayer {
    public ConvolutionLayerIgniteImpl(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        INDArray weights = getParam(ConvolutionParamInitializer.WEIGHT_KEY);

        int miniBatch = input.size(0);
        int inH = input.size(2);
        int inW = input.size(3);

        int outDepth = weights.size(0);
        int inDepth = weights.size(1);
        int kH = weights.size(2);
        int kW = weights.size(3);

        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();
        int[] pad;
        int[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, null, convolutionMode); //Also performs validation
            pad = ConvolutionUtils.getSameModeTopLeftPadding(outSize, new int[] {inH, inW}, kernel, strides);
        } else {
            pad = layerConf().getPadding();
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, pad, convolutionMode); //Also performs validation
        }

        int outH = outSize[0];
        int outW = outSize[1];


        INDArray biasGradView = gradientViews.get(ConvolutionParamInitializer.BIAS_KEY);
        INDArray weightGradView = gradientViews.get(ConvolutionParamInitializer.WEIGHT_KEY); //4d, c order. Shape: [outDepth,inDepth,kH,kW]
        INDArray weightGradView2df = Shape
                .newShapeNoCopy(weightGradView, new int[] {outDepth, inDepth * kH * kW}, false).transpose();



        INDArray delta;
        IActivation afn = layerConf().getActivationFn();

        Pair<INDArray, INDArray> p = preOutput4d(true, true);
        delta = afn.backprop(p.getFirst(), epsilon).getFirst(); //TODO handle activation function params


        delta = delta.permute(1, 0, 2, 3); //To shape: [outDepth,miniBatch,outH,outW]

        //Note: due to the permute in preOut, and the fact that we essentially do a preOut.muli(epsilon), this reshape
        // should be zero-copy; only possible exception being sometimes with the "identity" activation case
        INDArray delta2d = delta.reshape('c', new int[] {outDepth, miniBatch * outH * outW}); //Shape.newShapeNoCopy(delta,new int[]{outDepth,miniBatch*outH*outW},false);

        //Do im2col, but with order [miniB,outH,outW,depthIn,kH,kW]; but need to input [miniBatch,depth,kH,kW,outH,outW] given the current im2col implementation
        //To get this: create an array of the order we want, permute it to the order required by im2col implementation, and then do im2col on that
        //to get old order from required order: permute(0,3,4,5,1,2)
        INDArray im2col2d = p.getSecond(); //Re-use im2col2d array from forward pass if available; recalculate if not
        if (im2col2d == null) {
            INDArray col = Nd4j.createUninitialized(new int[] {miniBatch, outH, outW, inDepth, kH, kW}, 'c');
            INDArray col2 = col.permute(0, 3, 4, 5, 1, 2);
            Convolution.im2col(input, kH, kW, strides[0], strides[1], pad[0], pad[1],
                    convolutionMode == ConvolutionMode.Same, col2);
            //Shape im2col to 2d. Due to the permuting above, this should be a zero-copy reshape
            im2col2d = col.reshape('c', miniBatch * outH * outW, inDepth * kH * kW);
        }

        //Calculate weight gradients, using cc->c mmul.
        //weightGradView2df is f order, but this is because it's transposed from c order
        //Here, we are using the fact that AB = (B^T A^T)^T; output here (post transpose) is in c order, not usual f order
        Nd4j.gemm(im2col2d, delta2d, weightGradView2df, true, true, 1.0, 0.0);

        //Flatten 4d weights to 2d... this again is a zero-copy op (unless weights are not originally in c order for some reason)
        INDArray wPermuted = weights.permute(3, 2, 1, 0); //Start with c order weights, switch order to f order
        INDArray w2d = wPermuted.reshape('f', inDepth * kH * kW, outDepth);

        //Calculate epsilons for layer below, in 2d format (note: this is in 'image patch' format before col2im reduction)
        //Note: cc -> f mmul here, then reshape to 6d in f order
        INDArray epsNext2d = w2d.mmul(delta2d); //TODO can we reuse im2col array instead of allocating new result array?
        INDArray eps6d = Shape.newShapeNoCopy(epsNext2d, new int[] {kW, kH, inDepth, outW, outH, miniBatch}, true);

        //Calculate epsilonNext by doing im2col reduction.
        //Current col2im implementation expects input with order: [miniBatch,depth,kH,kW,outH,outW]
        //currently have [kH,kW,inDepth,outW,outH,miniBatch] -> permute first
        eps6d = eps6d.permute(5, 2, 1, 0, 4, 3);
        INDArray epsNextOrig = Nd4j.create(new int[] {inDepth, miniBatch, inH, inW}, 'c');


        //Note: we are execute col2im in a way that the output array should be used in a stride 1 muli in the layer below... (same strides as zs/activations)
        INDArray epsNext = epsNextOrig.permute(1, 0, 2, 3);
        Convolution.col2im(eps6d, epsNext, strides[0], strides[1], pad[0], pad[1], inH, inW);

        Gradient retGradient = new DefaultGradient();
        delta2d.sum(biasGradView, 1); //biasGradView is initialized/zeroed first in sum op

        retGradient.setGradientFor(ConvolutionParamInitializer.BIAS_KEY, biasGradView);
        retGradient.setGradientFor(ConvolutionParamInitializer.WEIGHT_KEY, weightGradView, 'c');

        return new Pair<>(retGradient, epsNext);
    }
}
