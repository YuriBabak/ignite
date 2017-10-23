package org.apache.ignite.ml.nn.util;

import java.util.Collection;
import org.apache.ignite.ml.math.Matrix;
import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.math.functions.Functions;
import org.apache.ignite.ml.math.functions.IgniteDoubleFunction;
import org.apache.ignite.ml.math.impls.matrix.DenseLocalOnHeapMatrix;
import org.apache.ignite.ml.math.impls.vector.DenseLocalOnHeapVector;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;


public class Algorithms {
    public static final IgniteDoubleFunction<Double> RELU = (x) -> Math.max(0.0, x);
    public static final IgniteDoubleFunction<Double> STEP = (x) -> (x > 0.0) ? 1.0 : 0.0;

    public static Matrix applyTo(String opName, Matrix data) {
        IgniteDoubleFunction<Double> unaryFun;
        if ("identity".equals(opName)) {
            unaryFun = Functions.IDENTITY;
        } else if ("relu".equals(opName)) {
            unaryFun = Algorithms.RELU;
        } else if ("step".equals(opName)) {
            unaryFun = Algorithms.STEP;
        } else {
            throw new RuntimeException("Unsupported activation function");
        }
        return data.map(unaryFun);
    }
    public static Tensor4d applyTo(String opName, Tensor4d tensor) {
        for (int i = 0; i != tensor.dims[0]; ++i) {
            for (int j = 0; j != tensor.dims[1]; ++j) {
                tensor.data[i][j] = applyTo(opName, tensor.data[i][j]);
            }
        }
        return tensor;
    }

    public static int convOutSize(int imageSize, int kernelSize, int stride, int padding) {
        return 1 + (imageSize + 2 * padding - kernelSize) / stride;
    }

    public static int deconvOutSize(int imageSize, int kernelSize, int stride, int padding) {
        return stride * (imageSize - 1) + kernelSize - 2 * padding;
    }

    /* Batch-generalization of im2col operation in order to zip the input 4d-tensor to 2d-matrix. */
    public static Matrix batch2col(Tensor4d batch, int kernel, int stride, int padding) {
        int batchSize = batch.dims[0];
        int depth = batch.dims[1];  // number of channels
        int[] imgShape = new int[]{ batch.dims[2], batch.dims[3] };
        int[] outputShape = new int[]{ convOutSize(imgShape[0], kernel, stride, padding),
                                       convOutSize(imgShape[1], kernel, stride, padding) };


        Matrix target = new DenseLocalOnHeapMatrix(depth * kernel * kernel,
                batchSize * outputShape[0] * outputShape[1]);
        int step = outputShape[0] * outputShape[1];
        for (int b = 0; b != batchSize; ++b) {
            for (int k = 0; k != depth * kernel * kernel * step; ++k) {
                int p = k / step;
                int q = k % step;

                int d0 = p / (kernel * kernel);
                int i0 = (q / outputShape[0]) + ((p / kernel) % kernel);
                int j0 = (q % outputShape[0]) + (p % kernel);

                if (padding <= i0 && i0 < imgShape[1] + padding &&
                        padding <= j0 && j0 < imgShape[0] + padding) {
                    double pixel = batch.data[b][d0].get(i0 - padding, j0 - padding);

                    target.set(p, b * step + q, pixel);
                }
            }
        }

        return target;
    }

    /* Batch generalization of col2im operation in order to unzip the 2d-matrix to 4d-tensor. */
    public static Tensor4d col2batch(Matrix dat2d, int[] targetShape, int kernel, int stride, int padding) {
        int batchSize = targetShape[0];
        int depth = targetShape[1];
        int[] imgShape = new int[]{ targetShape[2], targetShape[3] };

        int[] outputShape = new int[]{ convOutSize(imgShape[0], kernel, stride, padding),
                                       convOutSize(imgShape[1], kernel, stride, padding) };

        Tensor4d batch = new Tensor4d();
        batch.data = new Matrix[batchSize][depth];
        for (int b = 0; b != batchSize; ++b) {
            for (int d = 0; d != depth; ++d) {
                batch.data[b][d] = new DenseLocalOnHeapMatrix(imgShape[0], imgShape[1]);
            }
        }
        batch.dims = targetShape;


        int step = outputShape[0] * outputShape[1];
        for (int b = 0; b != batchSize; ++b) {
            for (int row = 0; row != depth * kernel * kernel; ++row) {
                int colOffset = row % kernel;
                int rowOffset = (row / kernel) % kernel;
                int d = row / (kernel * kernel);

                for (int h = 0; h != outputShape[0]; ++h) {
                    for (int w = 0; w != outputShape[1]; ++w) {
                        int x = rowOffset + h * stride - padding;
                        int y = colOffset + w * stride - padding;

                        double val = dat2d.get(row, b * step + h * outputShape[1] + w);
                        if (x < 0 || y < 0 || x >= imgShape[0] || y >= imgShape[1]) {
                            continue;
                        }
                        double val0 = batch.data[b][d].get(x, y);
                        batch.data[b][d].set(x, y, val);
                    }
                }
            }
        }

        return batch;
    }


    public static INDArray create(int rows, int cols) {
        return Nd4j.create(rows, cols);
    }

    public static INDArray valueArrayOf(int num, double value) {
        return Nd4j.valueArrayOf(new int[]{1, num}, value);
    }

    public static INDArray ones(int[] shape) {
        return Nd4j.ones(shape);
    }

    public static INDArray toFlattened(Collection<INDArray> matrices) {
        int length = 0;
        for (INDArray m : matrices)
            length += m.length();
        INDArray ret = create(1, length);
        int linearIndex = 0;
        for (INDArray d : matrices) {
            ret.put(new INDArrayIndex[] {NDArrayIndex.interval(linearIndex, linearIndex + d.length())}, d);
            linearIndex += d.length();
        }

        return ret;
    }

    public static INDArray toNd4j(Matrix matrix) {
        INDArray m = create(matrix.rowSize(), matrix.columnSize());

        for (int i = 0; i != matrix.rowSize(); ++i) {
            for (int j = 0; j != matrix.columnSize(); ++j) {
                m.putScalar(i, j, matrix.get(i, j));
            }
        }

        return m;
    }
    public static INDArray toNd4j(Vector vector) {
        INDArray m = create(vector.size(), 1);

        for (int i = 0; i != vector.size(); ++i) {
            m.putScalar(i, 0, vector.get(i));
        }

        return m;
    }

    public static Matrix toIgnite(INDArray matrix) {
        assert(matrix.rank() == 2);
        Matrix m = new DenseLocalOnHeapMatrix(matrix.rows(), matrix.columns());

        for (int i = 0; i != matrix.rows(); ++i) {
            for (int j = 0; j != matrix.columns(); ++j) {
                m.set(i, j, matrix.getDouble(i, j));
            }
        }

        return m;
    }

    public static Matrix hadamardProduct(Matrix m1, Matrix m2) {
        assert(m1.rowSize() == m2.rowSize() && m1.columnSize() == m2.columnSize());

        Matrix product = new DenseLocalOnHeapMatrix(m1.rowSize(), m1.columnSize());
        for (int i = 0; i != m1.rowSize(); ++i) {
            for (int j = 0; j != m1.columnSize(); ++j) {
                product.set(i, j, m1.get(i, j) * m2.get(i, j));
            }
        }

        return product;
    }

    public static Vector vec(Matrix m) {
        Vector stacked = new DenseLocalOnHeapVector(m.rowSize() * m.columnSize());

        int step = m.rowSize();
        for (int i = 0; i != m.columnSize(); ++i) {
            Vector col = m.viewColumn(i);

            stacked.viewPart(i * step, step).assign(col);
        }

        return stacked;
    }

    public static Matrix unvec(Vector v, int rows) {
        Matrix unrolled = new DenseLocalOnHeapMatrix(rows, v.size() / rows);

        for (int i = 0; i != v.size() / rows; ++i) {
            Vector col = v.viewPart(i * rows, rows);

            unrolled.viewColumn(i).assign(col);
        }
        return unrolled;
    }

    public static Vector sumRows(Matrix m) {
        Vector acc = new DenseLocalOnHeapVector(m.columnSize());

        for (int i = 0; i != m.columnSize(); ++i) {
            acc = acc.plus(m.viewRow(i));
        }
        return acc;
    }
}
