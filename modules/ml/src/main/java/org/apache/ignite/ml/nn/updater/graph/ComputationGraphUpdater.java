package org.apache.ignite.ml.nn.updater.graph;

import java.util.HashMap;
import java.util.Map;
import org.apache.ignite.ml.nn.api.Layer;
import org.apache.ignite.ml.nn.api.Updater;
import org.apache.ignite.ml.nn.gradient.DefaultGradient;
import org.apache.ignite.ml.nn.gradient.Gradient;
import org.apache.ignite.ml.nn.graph.ComputationGraph;
import org.apache.ignite.ml.nn.updater.UpdaterCreator;
import org.nd4j.linalg.api.ndarray.INDArray;


public class ComputationGraphUpdater implements Cloneable {

    private final Updater[] layerUpdaters;
    private final Map<String,Integer> layerUpdatersMap;

    public ComputationGraphUpdater(ComputationGraph graph){
        layerUpdaters = new Updater[graph.getNumLayers()];
        layerUpdatersMap = new HashMap<>();

        int i=0;
        for(Layer layer : graph.getLayers()){
            Updater u = UpdaterCreator.getUpdater(layer);
            layerUpdaters[i] = u;
            layerUpdatersMap.put(layer.conf().getLayer().getLayerName(),i);
            i++;
        }
    }

    private ComputationGraphUpdater(ComputationGraphUpdater updater){
        layerUpdaters = new Updater[updater.layerUpdaters.length];
        for( int i=0; i<layerUpdaters.length; i++ ) layerUpdaters[i] = updater.layerUpdaters[i].clone();
        layerUpdatersMap = new HashMap<>(updater.layerUpdatersMap);
    }

    @Override
    public ComputationGraphUpdater clone(){
        return new ComputationGraphUpdater(this);
    }

    public void update(ComputationGraph graph, Gradient gradient, int iteration, int batchSize ){
        Map<String,Gradient> layerGradients = new HashMap<>();

        for(Map.Entry<String,INDArray> gradientPair : gradient.gradientForVariable().entrySet()) {
            String key = gradientPair.getKey();
            int idx = key.lastIndexOf('_');
            if( idx == -1 ) throw new IllegalStateException("Invalid key: ComputationGraph Gradient key does not have layer separator: \""+key+"\"");

            String layerName = key.substring(0,idx);

            Gradient g = layerGradients.get(layerName);
            if(g == null){
                g = new DefaultGradient();
                layerGradients.put(layerName,g);
            }

            String newKey = key.substring(idx + 1);
            g.setGradientFor(newKey, gradientPair.getValue());
        }

        for(Map.Entry<String,Gradient> entry : layerGradients.entrySet() ){
            String layerName = entry.getKey();
            int updaterIdx = layerUpdatersMap.get(layerName);
            layerUpdaters[updaterIdx].update(graph.getLayer(layerName),entry.getValue(),iteration,batchSize);

            for( Map.Entry<String, INDArray> entry2 : layerGradients.get(layerName).gradientForVariable().entrySet() ){
                gradient.setGradientFor(entry.getKey()+"_"+entry2.getKey(), entry2.getValue());
            }
        }
    }
}
