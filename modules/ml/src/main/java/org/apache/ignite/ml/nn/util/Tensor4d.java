package org.apache.ignite.ml.nn.util;

import org.apache.ignite.ml.math.Matrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


public class Tensor4d {
    public Matrix[][] data;
    public int[] dims;

    public INDArray toNd4j() {
        INDArray m = Nd4j.create(dims);

        for (int i = 0; i != this.dims[0]; ++i) {
            for (int j = 0; j != this.dims[1]; ++j) {
                m.slice(i).slice(j).assign(Algorithms.toNd4j(this.data[i][j]));
            }
        }

        return m;
    }

    public void muli(Tensor4d other) {
        for (int i = 0; i != this.dims[0]; ++i) {
            for (int j = 0; j != this.dims[1]; ++j) {
                Matrix slice = this.data[i][j];

                slice.assign(Algorithms.hadamardProduct(slice, other.data[i][j]));
            }
        }
    }

    public static Tensor4d toIgnite(INDArray array) {
        assert(array.rank() == 4);

        Tensor4d tensor = new Tensor4d();
        tensor.data = new Matrix[array.size(0)][array.size(1)];
        tensor.dims = array.shape();

        for (int i = 0; i != tensor.dims[0]; ++i) {
            for (int j = 0; j != tensor.dims[1]; ++j) {
                INDArray xs = array.slice(i).slice(j);

                tensor.data[i][j] = Algorithms.toIgnite(xs);
            }
        }

        return tensor;
    }
}
