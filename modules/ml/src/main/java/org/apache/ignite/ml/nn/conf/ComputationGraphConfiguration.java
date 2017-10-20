package org.apache.ignite.ml.nn.conf;

import lombok.*;
import org.apache.ignite.ml.nn.conf.graph.GraphVertex;
import org.apache.ignite.ml.nn.conf.graph.LayerVertex;
import org.apache.ignite.ml.nn.conf.inputs.InputType;
import org.apache.ignite.ml.nn.conf.layers.*;
import org.apache.ignite.ml.nn.conf.preprocessor.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;


@Data
@EqualsAndHashCode
@AllArgsConstructor(access = AccessLevel.PRIVATE)
@NoArgsConstructor
public class ComputationGraphConfiguration implements Cloneable {
    private static Logger log = LoggerFactory.getLogger(ComputationGraphConfiguration.class);

    protected Map<String, GraphVertex> vertices = new LinkedHashMap<>();
    protected Map<String, List<String>> vertexInputs = new LinkedHashMap<>();

    protected List<String> networkInputs;

    protected List<String> networkOutputs;

    protected NeuralNetConfiguration defaultConfiguration;

    @Override
    public ComputationGraphConfiguration clone() {
        ComputationGraphConfiguration conf = new ComputationGraphConfiguration();

        conf.vertices = new HashMap<>();
        for (Map.Entry<String, GraphVertex> entry : this.vertices.entrySet()) {
            conf.vertices.put(entry.getKey(), entry.getValue().clone());
        }

        conf.vertexInputs = new HashMap<>();
        for (Map.Entry<String, List<String>> entry : this.vertexInputs.entrySet()) {
            conf.vertexInputs.put(entry.getKey(), new ArrayList<>(entry.getValue()));
        }
        conf.networkInputs = new ArrayList<>(this.networkInputs);
        conf.networkOutputs = new ArrayList<>(this.networkOutputs);

        conf.defaultConfiguration = defaultConfiguration.clone();

        return conf;
    }


    public void addPreProcessors(InputType... inputTypes) {

        if (inputTypes == null || inputTypes.length != networkInputs.size()) {
            throw new IllegalArgumentException("Invalid number of InputTypes: cannot add preprocessors if number of InputType "
                    + "objects differs from number of network inputs");
        }

        Map<String, List<String>> verticesOutputTo = new HashMap<>();
        for (Map.Entry<String, GraphVertex> entry : vertices.entrySet()) {
            String vertexName = entry.getKey();
            List<String> vertexInputNames;
            vertexInputNames = vertexInputs.get(vertexName);

            if (vertexInputNames == null) continue;

            for (String s : vertexInputNames) {
                List<String> list = verticesOutputTo.get(s);
                if (list == null) {
                    list = new ArrayList<>();
                    verticesOutputTo.put(s, list);
                }
                list.add(vertexName);   //Edge: s -> vertexName
            }
        }

        LinkedList<String> noIncomingEdges = new LinkedList<>(networkInputs);
        List<String> topologicalOrdering = new ArrayList<>();

        Map<String, Set<String>> inputEdges = new HashMap<>();
        for (Map.Entry<String, List<String>> entry : vertexInputs.entrySet()) {
            inputEdges.put(entry.getKey(), new HashSet<>(entry.getValue()));
        }

        while (!noIncomingEdges.isEmpty()) {
            String next = noIncomingEdges.removeFirst();
            topologicalOrdering.add(next);

            List<String> nextEdges = verticesOutputTo.get(next);

            if (nextEdges != null && !nextEdges.isEmpty()) {
                for (String s : nextEdges) {
                    Set<String> set = inputEdges.get(s);
                    set.remove(next);
                    if (set.isEmpty()) {
                        noIncomingEdges.add(s);
                    }
                }
            }
        }

        for (Map.Entry<String, Set<String>> entry : inputEdges.entrySet()) {
            Set<String> set = entry.getValue();
            if (set == null) continue;
            if (!set.isEmpty())
                throw new IllegalStateException("Invalid configuration: cycle detected in graph. Cannot calculate topological ordering with graph cycle ("
                        + "cycle includes vertex \"" + entry.getKey() + "\")");
        }

        Map<String, InputType> vertexOutputs = new HashMap<>();
        for (String s : topologicalOrdering) {
            int inputIdx = networkInputs.indexOf(s);
            if (inputIdx != -1) {
                vertexOutputs.put(s, inputTypes[inputIdx]);
                continue;
            }
            GraphVertex gv = vertices.get(s);

            List<InputType> inputTypeList = new ArrayList<>();

            if (gv instanceof LayerVertex) {
                String in = vertexInputs.get(s).get(0);
                InputType layerInput = vertexOutputs.get(in);

                LayerVertex lv = (LayerVertex) gv;
                if (lv.getPreProcessor() != null) continue;

                Layer l = lv.getLayerConf().getLayer();
                if (l instanceof ConvolutionLayer) {
                    switch (layerInput.getType()) {
                        case FF:
                            log.warn("Automatic addition of FF -> CNN preprocessors: not yet implemented (layer: " + s + ")");
                            break;
                        case CNN:
                            if(networkInputs.contains(vertexInputs.get(s).get(0))){
                                InputType.InputTypeConvolutional conv = (InputType.InputTypeConvolutional) layerInput;
                                lv.setPreProcessor(new FeedForwardToCnnPreProcessor(conv.getHeight(), conv.getWidth(), conv.getDepth()));
                            }
                            break;
                    }
                } else {
                    switch (layerInput.getType()) {
                        case FF:
                            setNInIfNecessary(lv,layerInput);
                            break;
                        case CNN:
                            InputType.InputTypeConvolutional conv = (InputType.InputTypeConvolutional) layerInput;
                            lv.setPreProcessor(new CnnToFeedForwardPreProcessor(conv.getHeight(), conv.getWidth(), conv.getDepth()));
                            int nIn = conv.getHeight() * conv.getWidth() * conv.getDepth();
                            ((FeedForwardLayer) lv.getLayerConf().getLayer()).setNIn(nIn);
                            break;
                    }
                }
                inputTypeList.add(layerInput);
            } else {
                List<String> inputs = vertexInputs.get(s);
                if (inputs != null) {
                    for (String inputVertexName : inputs) {
                        inputTypeList.add(vertexOutputs.get(inputVertexName));
                    }
                }
            }

            InputType outputFromVertex = gv.getOutputType(inputTypeList.toArray(new InputType[inputTypeList.size()]));
            vertexOutputs.put(s, outputFromVertex);
        }
    }

    private static void setNInIfNecessary(LayerVertex lv, InputType inputType){
        FeedForwardLayer ffl = (FeedForwardLayer) lv.getLayerConf().getLayer();
        if(ffl.getNIn() == 0){
            int size;
            if(inputType instanceof InputType.InputTypeFeedForward){
                size = ((InputType.InputTypeFeedForward) inputType).getSize();
            } else throw new UnsupportedOperationException("Invalid input type");
            if(size > 0) ffl.setNIn(size);
        }
    }


    @Data
    public static class GraphBuilder {
        protected Map<String, GraphVertex> vertices = new LinkedHashMap<>();

        protected Map<String, List<String>> vertexInputs = new LinkedHashMap<>();

        protected List<String> networkInputs = new ArrayList<>();
        protected List<InputType> networkInputTypes = new ArrayList<>();
        protected List<String> networkOutputs = new ArrayList<>();

        protected Map<String, InputPreProcessor> inputPreProcessors = new LinkedHashMap<>();

        protected NeuralNetConfiguration.Builder globalConfiguration;


        public GraphBuilder(NeuralNetConfiguration.Builder globalConfiguration) {
            this.globalConfiguration = globalConfiguration;
        }

        public GraphBuilder addLayer(String layerName, Layer layer, String... layerInputs) {
            return addLayer(layerName, layer, null, layerInputs);
        }

        public GraphBuilder addLayer(String layerName, Layer layer, InputPreProcessor preProcessor, String... layerInputs) {
            NeuralNetConfiguration.Builder builder = globalConfiguration.clone();
            builder.layer(layer);
            vertices.put(layerName, new LayerVertex(builder.build(), preProcessor));

            if (layerInputs != null) {
                this.vertexInputs.put(layerName, Arrays.asList(layerInputs));
            }
            layer.setLayerName(layerName);
            return this;
        }

        public GraphBuilder addInputs(String... inputNames) {
            Collections.addAll(networkInputs, inputNames);
            return this;
        }

        public GraphBuilder setInputTypes(InputType... inputTypes) {
            if(inputTypes != null && inputTypes.length > 0) Collections.addAll(networkInputTypes, inputTypes);
            return this;
        }

        public GraphBuilder setOutputs(String... outputNames) {
            Collections.addAll(networkOutputs, outputNames);
            return this;
        }

        public GraphBuilder addVertex(String vertexName, GraphVertex vertex, String... vertexInputs) {
            vertices.put(vertexName, vertex);
            this.vertexInputs.put(vertexName, Arrays.asList(vertexInputs));
            return this;
        }

        public ComputationGraphConfiguration build() {

            ComputationGraphConfiguration conf = new ComputationGraphConfiguration();

            conf.networkInputs = networkInputs;
            conf.networkOutputs = networkOutputs;

            conf.vertices = this.vertices;
            conf.vertexInputs = this.vertexInputs;

            conf.defaultConfiguration = globalConfiguration.build();

            for (Map.Entry<String, InputPreProcessor> entry : inputPreProcessors.entrySet()) {
                GraphVertex gv = vertices.get(entry.getKey());
                if (gv instanceof LayerVertex) {
                    LayerVertex lv = (LayerVertex) gv;
                    lv.setPreProcessor(entry.getValue());
                } else {
                    throw new IllegalStateException("Invalid configuration: InputPreProcessor defined for GraphVertex \"" + entry.getKey()
                            + "\", but this vertex is not a LayerVertex");
                }
            }

            if (!networkInputTypes.isEmpty()) {
                conf.addPreProcessors(networkInputTypes.toArray(new InputType[networkInputs.size()]));
            }

            return conf;
        }
    }
}
