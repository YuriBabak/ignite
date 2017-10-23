package org.apache.ignite.ml.nn.api;

import java.util.Map;
import org.apache.ignite.lang.IgniteBiTuple;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.apache.ignite.ml.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;


public interface Model {
    double score();

    void computeGradientAndScore();

    INDArray params();

    int numParams();

    int numParams(boolean backwards);

    void setParams(INDArray params);

    void setParamsViewArray(INDArray params);

    void setBackpropGradientsViewArray(INDArray gradients);

    Gradient gradient();

    IgniteBiTuple<Gradient,Double> gradientAndScore();

    int batchSize();

    NeuralNetConfiguration conf();

    void setConf(NeuralNetConfiguration conf);

    INDArray input();

    INDArray getParam(String param);

    Map<String,INDArray> paramTable();

    void setParamTable(Map<String,INDArray> paramTable);
}
