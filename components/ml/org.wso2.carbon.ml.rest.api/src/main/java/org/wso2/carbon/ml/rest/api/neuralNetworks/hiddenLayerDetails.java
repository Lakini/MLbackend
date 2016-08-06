package org.wso2.carbon.ml.rest.api.neuralNetworks;

/**
 * Created by Lakini on 8/6/2016.
 */
public class HiddenLayerDetails {
    int hiddenNodes;
    String weightInit;
    String activationAlgo;

    public HiddenLayerDetails(int hiddenNodes, String weightInit, String activationAlgo) {
        this.hiddenNodes = hiddenNodes;
        this.weightInit = weightInit;
        this.activationAlgo = activationAlgo;
    }
}
