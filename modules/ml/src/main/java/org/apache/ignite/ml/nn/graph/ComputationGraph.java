package org.apache.ignite.ml.nn.graph;

import lombok.Setter;
import org.apache.ignite.internal.util.typedef.T3;
import org.apache.ignite.lang.IgniteBiTuple;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.apache.ignite.ml.nn.api.Layer;
import org.apache.ignite.ml.nn.api.Model;
import org.apache.ignite.ml.nn.conf.ComputationGraphConfiguration;
import org.apache.ignite.ml.nn.conf.NeuralNetConfiguration;
import org.apache.ignite.ml.nn.gradient.DefaultGradient;
import org.apache.ignite.ml.nn.gradient.Gradient;
import org.apache.ignite.ml.nn.graph.vertex.*;
import org.apache.ignite.ml.nn.graph.vertex.impl.*;
import org.apache.ignite.ml.nn.layers.BaseOutputLayer;
import org.apache.ignite.ml.nn.optimize.Solver;
import org.apache.ignite.ml.nn.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.ignite.ml.math.Matrix;
import org.apache.ignite.ml.nn.util.Algorithms;

import java.util.*;


public class ComputationGraph implements Model {
    private static final Logger log = LoggerFactory.getLogger(ComputationGraph.class);

    protected ComputationGraphConfiguration configuration;
    protected boolean initCalled = false;
    protected Solver solver;
    protected INDArray flattenedParams;
    protected INDArray flattenedGradients;
    protected Gradient gradient;
    protected double score;
    @Setter private boolean initDone = false;

    protected GraphVertex[] vertices;
    protected Map<String,GraphVertex> verticesMap;
    protected int[] topologicalOrder;
    protected Layer[] layers;

    private int numInputArrays;
    private int numOutputArrays;

    private INDArray[] inputs;
    private INDArray[] labels;

    private NeuralNetConfiguration defaultConfiguration;
    private Collection<IterationListener> listeners = new ArrayList<>();


    public ComputationGraph(ComputationGraphConfiguration configuration){
        this.configuration = configuration;
        this.numInputArrays = configuration.getNetworkInputs().size();
        this.numOutputArrays = configuration.getNetworkOutputs().size();
        this.inputs = new INDArray[numInputArrays];
        this.labels = new INDArray[numOutputArrays];
        this.defaultConfiguration = configuration.getDefaultConfiguration();
    }

    public ComputationGraphConfiguration getConfiguration(){
        return configuration;
    }

    public int getNumLayers(){
        return (layers != null ? layers.length : 0);
    }

    public Layer getLayer(int idx){
        return layers[idx];
    }

    public Layer[] getLayers(){
        return layers;
    }

    public Layer getLayer(String name){
        return verticesMap.get(name).getLayer();    //TODO checks
    }

    public void setInput(int inputNum, INDArray input){
        inputs[inputNum] = input;
    }

    public void setInputs(INDArray... inputs){
        if(inputs != null && inputs.length != this.numInputArrays){
            throw new IllegalArgumentException("Invalid input array: network has " + numInputArrays + " inputs, but array is of length " + inputs.length);
        }
        this.inputs = inputs;
    }

    public INDArray[] getInputs(){
        return inputs;
    }

    public void setLabel(int labelNum, INDArray label){
        labels[labelNum] = label;
    }

    public void setLabels(INDArray[] labels){
        if(labels != null && labels.length != this.numOutputArrays){
            throw new IllegalArgumentException("Invalid output array: network has " + numOutputArrays + " outputs, but array is of length " + labels.length);
        }
        this.labels = labels;
    }

    public void init() {
        init(null, false);
    }

    public void init(INDArray parameters, boolean cloneParametersArray){
        if(initCalled) return;

        topologicalOrder = topologicalSortOrder();

        Map<String,org.apache.ignite.ml.nn.conf.graph.GraphVertex> configVertexMap = configuration.getVertices();

        List<String> networkInputNames = configuration.getNetworkInputs();

        Map<String,List<String>> vertexInputs = configuration.getVertexInputs();
        this.vertices = new GraphVertex[networkInputNames.size() + configuration.getVertices().size()];

        Map<String,Integer> allNamesReverse = new HashMap<>();

        int vertexNumber=0;
        for( String name : networkInputNames){
            GraphVertex gv = new InputVertex(this,name,vertexNumber,null);  //Output vertices: set later
            allNamesReverse.put(name,vertexNumber);
            vertices[vertexNumber++] = gv;
        }

        int numParams = 0;
        int[] numParamsForVertex = new int[topologicalOrder.length];
        int i=0;
        for(; i<configuration.getNetworkInputs().size(); i++ ){
            numParamsForVertex[i] = 0;  //No parameters for input vertices
        }
        for(Map.Entry<String,org.apache.ignite.ml.nn.conf.graph.GraphVertex> nodeEntry : configVertexMap.entrySet()){
            org.apache.ignite.ml.nn.conf.graph.GraphVertex n = nodeEntry.getValue();
            numParamsForVertex[i] = n.numParams(true);
            numParams += numParamsForVertex[i];
            i++;
        }

        boolean initializeParams;
        if(parameters != null){
            if(!parameters.isRowVector()) throw new IllegalArgumentException("Invalid parameters: should be a row vector");
            if(parameters.length() != numParams) throw new IllegalArgumentException("Invalid parameters: expected length " + numParams + ", got length " + parameters.length());

            if(cloneParametersArray) flattenedParams = parameters.dup();
            else flattenedParams = parameters;

            initializeParams = false;
        } else {
            flattenedParams = Algorithms.create(1,numParams);
            initializeParams = true;
        }

        INDArray[] paramsViewForVertex = new INDArray[topologicalOrder.length];
        int paramOffsetSoFar = 0;
        i=0;
        for( int vertexIdx : topologicalOrder ){
            int nParamsThisVertex = numParamsForVertex[vertexIdx];
            if(nParamsThisVertex != 0){
                paramsViewForVertex[vertexIdx] = flattenedParams.get(NDArrayIndex.point(0), NDArrayIndex.interval(paramOffsetSoFar, paramOffsetSoFar + nParamsThisVertex));
            }
            i++;
            paramOffsetSoFar += nParamsThisVertex;
        }


        int numLayers = 0;
        List<Layer> tempLayerList = new ArrayList<>();
        for( Map.Entry<String,org.apache.ignite.ml.nn.conf.graph.GraphVertex> nodeEntry : configVertexMap.entrySet() ){
            org.apache.ignite.ml.nn.conf.graph.GraphVertex n = nodeEntry.getValue();
            String name = nodeEntry.getKey();
            GraphVertex gv = n.instantiate(this,name,vertexNumber,paramsViewForVertex[vertexNumber], initializeParams);

            if(gv.hasLayer()){
                numLayers++;
                tempLayerList.add(gv.getLayer());
            }

            allNamesReverse.put(name,vertexNumber);
            vertices[vertexNumber++] = gv;
        }
        layers = tempLayerList.toArray(new Layer[numLayers]);

        verticesMap = new HashMap<>();
        for(GraphVertex gv : vertices){
            verticesMap.put(gv.getVertexName(),gv);
        }

        Map<String,List<String>> verticesOutputTo = new HashMap<>();
        for( GraphVertex gv : vertices ){
            String vertexName = gv.getVertexName();
            List<String> vertexInputNames;
            vertexInputNames = vertexInputs.get(vertexName);

            if(vertexInputNames == null) continue;

            for(String s : vertexInputNames){
                List<String> list = verticesOutputTo.get(s);
                if(list == null){
                    list = new ArrayList<>();
                    verticesOutputTo.put(s,list);
                }
                list.add(vertexName);
            }
        }


        for( GraphVertex gv : vertices ){
            String vertexName = gv.getVertexName();
            int vertexIndex = gv.getVertexIndex();
            List<String> vertexInputNames;
            vertexInputNames = vertexInputs.get(vertexName);

            if(vertexInputNames == null) continue;

            VertexIndices[] inputIndices = new VertexIndices[vertexInputNames.size()];
            for( int j=0; j<vertexInputNames.size(); j++ ){
                String inName = vertexInputNames.get(j);
                int inputVertexIndex = allNamesReverse.get(inName);

                GraphVertex inputVertex = vertices[inputVertexIndex];
                List<String> inputVertexOutputsTo = verticesOutputTo.get(inName);
                int outputNumberOfInput = inputVertexOutputsTo.indexOf(vertexName);


                if(outputNumberOfInput == -1) throw new IllegalStateException("Could not find vertex " + vertexIndex + " in the list of outputs "
                    + "for vertex " + inputVertex + "; error in graph structure?");

                inputIndices[j] = new VertexIndices(inputVertexIndex,outputNumberOfInput);
            }

            gv.setInputVertices(inputIndices);
        }

        for( GraphVertex gv : vertices ) {
            String vertexName = gv.getVertexName();

            List<String> thisVertexOutputsTo = verticesOutputTo.get(vertexName);

            if(thisVertexOutputsTo == null || thisVertexOutputsTo.isEmpty()) continue;
            VertexIndices[] outputIndices = new VertexIndices[thisVertexOutputsTo.size()];
            int j=0;
            for( String s : thisVertexOutputsTo ){
                List<String> nextVertexInputNames = vertexInputs.get(s);

                int outputVertexInputNumber = nextVertexInputNames.indexOf(vertexName);

                int outputVertexIndex = allNamesReverse.get(s);
                outputIndices[j++] = new VertexIndices(outputVertexIndex,outputVertexInputNumber);
            }
            gv.setOutputVertices(outputIndices);
        }

        initCalled = true;
    }

    public void initGradientsView(){
        if(!initCalled) init();

        int numParams = 0;
        int[] numParamsForVertex = new int[topologicalOrder.length];
        int i=0;
        for(; i<configuration.getNetworkInputs().size(); i++ ){
            numParamsForVertex[i] = 0;  //No parameters for input vertices
        }
        Map<String,org.apache.ignite.ml.nn.conf.graph.GraphVertex> configVertexMap = configuration.getVertices();
        for(Map.Entry<String,org.apache.ignite.ml.nn.conf.graph.GraphVertex> nodeEntry : configVertexMap.entrySet()){
            org.apache.ignite.ml.nn.conf.graph.GraphVertex n = nodeEntry.getValue();
            numParamsForVertex[i] = n.numParams(true);
            numParams += numParamsForVertex[i];
            i++;
        }
        flattenedGradients = Algorithms.create(1,numParams);

        int paramOffsetSoFar = 0;
        i=0;
        for( int vertexIdx : topologicalOrder ){
            int nParamsThisVertex = numParamsForVertex[vertexIdx];
            if(nParamsThisVertex != 0){
                INDArray gradientView = flattenedGradients.get(NDArrayIndex.point(0), NDArrayIndex.interval(paramOffsetSoFar, paramOffsetSoFar + nParamsThisVertex));
                vertices[vertexIdx].setBackpropGradientsViewArray(gradientView);
            }
            i++;
            paramOffsetSoFar += nParamsThisVertex;
        }


    }

    public void fit(DataSetIterator iterator){
        DataSetIterator dataSetIterator = iterator;

        while (dataSetIterator.hasNext()) {
            DataSet next = dataSetIterator.next();
            if (next.getFeatureMatrix() == null || next.getLabels() == null) {
                break;
            }

            Matrix features = Algorithms.toIgnite(next.getFeatureMatrix());
            Matrix labels = Algorithms.toIgnite(next.getLabels());

            setInput(0, next.getFeatureMatrix());
            setLabel(0, next.getLabels());
            if( solver == null ){
                solver = new Solver.Builder()
                        .configure(defaultConfiguration)    //TODO; don't like this
                        .listeners(listeners)
                        .model(this).build();
            }
            solver.optimize();
        }
    }

    public int[] topologicalSortOrder(){
        if(topologicalOrder != null) return topologicalOrder;

        Map<String,org.apache.ignite.ml.nn.conf.graph.GraphVertex> nodeMap = configuration.getVertices();
        List<String> networkInputNames = configuration.getNetworkInputs();
        int numVertices = networkInputNames.size() + configuration.getVertices().size();
        int[] out = new int[numVertices];
        int outCounter = 0;

        Map<Integer,String> vertexNamesMap = new HashMap<>();
        Map<String,Integer> vertexNamesMap2 = new HashMap<>();
        int i=0;
        for( String inputName : configuration.getNetworkInputs()){
            vertexNamesMap.put(i,inputName);
            vertexNamesMap2.put(inputName, i);
            i++;
        }
        for( Map.Entry<String,org.apache.ignite.ml.nn.conf.graph.GraphVertex> entry : nodeMap.entrySet()) {
            String name = entry.getKey();
            vertexNamesMap.put(i,name);
            vertexNamesMap2.put(name,i);
            i++;
        }

        Map<Integer,Set<Integer>> inputEdges = new HashMap<>();
        Map<Integer,Set<Integer>> outputEdges = new HashMap<>();

        for(String s : configuration.getNetworkInputs() ){
            int idx = vertexNamesMap2.get(s);
            inputEdges.put(idx, null);
        }

        for( Map.Entry<String,org.apache.ignite.ml.nn.conf.graph.GraphVertex> entry : nodeMap.entrySet()){
            String thisVertexName  = entry.getKey();
            int idx = vertexNamesMap2.get(thisVertexName);
            List<String> inputsToThisVertex = configuration.getVertexInputs().get(thisVertexName);

            if(inputsToThisVertex == null || inputsToThisVertex.isEmpty()){
                inputEdges.put(idx,null);
                continue;
            }

            Set<Integer> inputSet = new HashSet<>();
            for(String s : inputsToThisVertex){
                Integer inputIdx = vertexNamesMap2.get(s);
                if(inputIdx==null){
                    System.out.println();
                }
                inputSet.add(inputIdx);
                Set<Integer> outputSetForInputIdx = outputEdges.get(inputIdx);
                if(outputSetForInputIdx == null){
                    outputSetForInputIdx = new HashSet<>();
                    outputEdges.put(inputIdx,outputSetForInputIdx);
                }
                outputSetForInputIdx.add(idx);  //input vertex outputs to the current vertex
            }
            inputEdges.put(idx, inputSet);
        }

        LinkedList<Integer> noIncomingEdges = new LinkedList<>();
        for( Map.Entry<Integer,Set<Integer>> entry : inputEdges.entrySet() ) {
            Set<Integer> inputsFrom = entry.getValue();
            if(inputsFrom == null || inputsFrom.isEmpty()) {
                noIncomingEdges.add(entry.getKey());
            }
        }

        while(!noIncomingEdges.isEmpty()) {
            int next = noIncomingEdges.removeFirst();
            out[outCounter++] = next;

            Set<Integer> vertexOutputsTo = outputEdges.get(next);

            if(vertexOutputsTo != null ) {
                for( Integer v : vertexOutputsTo){
                    Set<Integer> set = inputEdges.get(v);
                    set.remove(next);
                    if (set.isEmpty()) {
                        noIncomingEdges.add(v);
                    }
                }
            }
        }

        for(Map.Entry<Integer,Set<Integer>> entry : inputEdges.entrySet()){
            Set<Integer> set = entry.getValue();
            if(set == null) continue;
            if(!set.isEmpty()) throw new IllegalStateException("Invalid configuration: cycle detected in graph. Cannot calculate topological ordering with graph cycle ("
                    + "cycle includes vertex \"" + vertexNamesMap.get(entry.getKey()) + "\")");
        }

        return out;
    }

    @Override
    public void computeGradientAndScore() {
        feedForward(true, true);
        backprop();

        double l1 = calcL1();
        double l2 = calcL2();

        score = 0.0;
        for(String s : configuration.getNetworkOutputs()){
            GraphVertex gv = verticesMap.get(s);

            score += ((BaseOutputLayer<?>)gv.getLayer()).computeScore(l1,l2);

            l1 = 0.0;
            l2 = 0.0;
        }
    }

    public Map<String,INDArray> feedForward(boolean train) {
        return feedForward(train, false);
    }

    private Map<String,INDArray> feedForward(boolean train, boolean excludeOutputLayers){
        Map<String,INDArray> layerActivations = new HashMap<>();

        for( int i=0; i<topologicalOrder.length; i++ ){
            GraphVertex current = vertices[topologicalOrder[i]];
            if(current.isInputVertex()){
                VertexIndices[] inputsTo = current.getOutputVertices();
                INDArray input = inputs[current.getVertexIndex()];

                layerActivations.put(current.getVertexName(),input);

                for( VertexIndices v : inputsTo ){
                    int vIdx = v.getVertexIndex();
                    int vIdxInputNum = v.getVertexEdgeNumber();
                    vertices[vIdx].setInput(vIdxInputNum,input.dup());
                }

            } else {
                if(excludeOutputLayers && current.isOutputVertex() && current.hasLayer() && current.getLayer() instanceof BaseOutputLayer){
                    continue;
                }
                INDArray out = current.doForward(train);

                if(current.hasLayer()){
                    layerActivations.put(current.getVertexName(),out);
                }

                VertexIndices[] outputsTo = current.getOutputVertices();
                if(outputsTo != null) {
                    for (VertexIndices v : outputsTo) {
                        int vIdx = v.getVertexIndex();
                        int inputNum = v.getVertexEdgeNumber();
                        vertices[vIdx].setInput(inputNum, out);
                    }
                }
            }
        }

        return layerActivations;
    }

    public INDArray[] output(INDArray... input){
        return output(false, input);
    }

    public INDArray[] output(boolean train, INDArray... input){
        setInputs(input);
        Map<String,INDArray> activations = feedForward(train);
        INDArray[] outputs = new INDArray[numOutputArrays];
        int i=0;
        for(String s : configuration.getNetworkOutputs()){
            outputs[i++] = activations.get(s);
        }
        return outputs;
    }

    protected void backprop(){
        if(flattenedGradients == null) initGradientsView();

        LinkedList<T3<String,INDArray,Character>> gradients = new LinkedList<>();

        for( int i=topologicalOrder.length-1; i>= 0; i-- ){
            GraphVertex current = vertices[topologicalOrder[i]];

            if(current.isInputVertex()) continue;   //No op

            if(current.isOutputVertex()){
                BaseOutputLayer<?> outputLayer = (BaseOutputLayer<?>)current.getLayer();

                int thisOutputNumber = configuration.getNetworkOutputs().indexOf(current.getVertexName());
                INDArray currLabels = labels[thisOutputNumber];
                outputLayer.setLabels(currLabels);
            }

            IgniteBiTuple<Gradient,INDArray[]> pair = current.doBackward();
            INDArray[] epsilons = pair.get2();

            VertexIndices[] inputVertices = current.getInputVertices();

            if(inputVertices != null ){
                int j=0;
                for(VertexIndices v : inputVertices){
                    GraphVertex gv = vertices[v.getVertexIndex()];
                    int outputNumberOfInputVertex = v.getVertexEdgeNumber();
                    gv.setError(outputNumberOfInputVertex,epsilons[j++]);
                }
            }

            if(pair.get1() != null){
                Gradient g = pair.get1();
                Map<String,INDArray> map = g.gradientForVariable();
                LinkedList<T3<String,INDArray,Character>> tempList = new LinkedList<>();
                for( Map.Entry<String,INDArray> entry : map.entrySet() ){
                    String origName = entry.getKey();
                    String newName = current.getVertexName() + "_" + origName;
                    tempList.addFirst(new T3<>(newName,entry.getValue(), g.flatteningOrderForVariable(origName)));
                }
                for(T3<String,INDArray,Character> t : tempList ) gradients.addFirst(t);
            }
        }

        Gradient gradient = new DefaultGradient(flattenedGradients);
        for(T3<String,INDArray,Character> t : gradients ){
            gradient.setGradientFor(t.get1(),t.get2(),t.get3());
        }

        this.gradient = gradient;
    }

    public double calcL2() {
        double l2 = 0.0;
        for(Layer l : layers){
            l2 += l.calcL2();
        }
        return l2;
    }

    public double calcL1() {
        double l1 = 0.0;
        for(Layer l : layers){
            l1 += l.calcL1();
        }
        return l1;
    }

    public void setListeners(Collection<IterationListener> listeners){
        this.listeners = listeners;
        if(layers == null) init();

        for( Layer l : layers){
            l.setListeners(listeners);
        }

        if(solver != null){
            solver.setListeners(listeners);
        }
    }

    public void setListeners(IterationListener... listeners){
        List<IterationListener> list = new ArrayList<>();
        Collections.addAll(list,listeners);
        setListeners(list);
    }

    public Collection<IterationListener> getListeners(){
        return listeners;
    }

    public INDArray params(boolean backwardOnly){
        if(backwardOnly) return flattenedParams;

        List<INDArray> list = new ArrayList<>(layers.length);
        for( int i=0; i<topologicalOrder.length; i++ ){
            if(!vertices[topologicalOrder[i]].hasLayer()) continue;

            Layer l = vertices[topologicalOrder[i]].getLayer();
            INDArray layerParams = l.params();
            if(layerParams != null) list.add(layerParams);    //may be null: subsampling etc layers
        }

//        return Nd4j.toFlattened('f', list);
        return Algorithms.toFlattened(list);
    }

    @Override
    public double score() {
        return score;
    }

    @Override
    public INDArray params() {
        return params(true);
    }

    @Override
    public int numParams() {
        return numParams(true);
    }

    @Override
    public int numParams(boolean backwards) {
        int nParams = 0;
        for (Layer layer : layers) {
            nParams += layer.numParams(backwards);
        }
        return nParams;
    }

    @Override
    public void setParams(INDArray params) {
        if(params == flattenedParams) return;

        if(this.flattenedParams != null && this.flattenedParams.length() == params.length()){
            this.flattenedParams.assign(params);
            return;
        }

        int idx = 0;
        for( int i=0; i<topologicalOrder.length; i++ ){
            if(!vertices[topologicalOrder[i]].hasLayer()) continue;

            Layer layer = vertices[topologicalOrder[i]].getLayer();
            int range = layer.numParams();
            if(range <= 0) continue;    //Some layers: no parameters (subsampling etc)
            INDArray get = params.get(NDArrayIndex.point(0),NDArrayIndex.interval(idx, range + idx));
            layer.setParams(get);
            idx += range;
        }
    }

    @Override
    public void setParamsViewArray(INDArray params) {
        throw new RuntimeException("Not yet implemented");
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray gradients) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Gradient gradient() {
        return gradient;
    }

    @Override
    public IgniteBiTuple<Gradient, Double> gradientAndScore() {
        return new IgniteBiTuple<>(gradient(),score());
    }

    @Override
    public int batchSize() {
        return inputs[0].size(0);
    }

    @Override
    public NeuralNetConfiguration conf() {
        return defaultConfiguration;
    }

    @Override
    public void setConf(NeuralNetConfiguration conf) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray input() {
        if(numInputArrays == 1) return (inputs != null ? inputs[0] : null);
        else throw new UnsupportedOperationException("Cannot return single input: ComputationGraph  has multiple inputs");
    }

    @Override
    public INDArray getParam(String param) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public Map<String, INDArray> paramTable() {
        Map<String,INDArray> allParams = new LinkedHashMap<>();
        for (Layer layer : layers) {
            Map<String, INDArray> paramMap = layer.paramTable();
            for (Map.Entry<String, INDArray> entry : paramMap.entrySet()) {
                String newKey = layer.conf().getLayer().getLayerName() + "_" + entry.getKey();
                allParams.put(newKey, entry.getValue());
            }
        }
        return allParams;
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        throw new UnsupportedOperationException("Not implemented");
    }
}
