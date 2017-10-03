package org.apache.ignite.ml.nn;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.DataSet;


@Data
@AllArgsConstructor
public class DistributionParams {
    protected ComputationGraph model = null;
    protected DataSet samples = null;
}
