package org.wso2.carbon.ml.rest.api.neuralNetworks;

/**
 * Created by Lakini on 8/6/2016.
 */
public class OutputLayerDetails {
    int outputNodes;
    String weightInit;
    String activationAlgo;
    String lossFunction;

    public OutputLayerDetails(int outputNodes, String weightInit, String activationAlgo, String lossFunction) {
        this.outputNodes = outputNodes;
        this.weightInit = weightInit;
        this.activationAlgo = activationAlgo;
        this.lossFunction = lossFunction;
    }
}
