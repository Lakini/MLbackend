/*
 * Copyright (c) 2015, WSO2 Inc. (http://www.wso2.org) All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.wso2.carbon.ml.rest.api.neuralNetworks;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.wso2.carbon.context.PrivilegedCarbonContext;
import org.wso2.carbon.ml.commons.domain.MLModelData;
import org.wso2.carbon.ml.core.exceptions.MLAnalysisHandlerException;
import org.wso2.carbon.ml.core.exceptions.MLDataProcessingException;
import org.wso2.carbon.ml.core.exceptions.MLModelHandlerException;
import org.wso2.carbon.ml.core.impl.MLAnalysisHandler;
import org.wso2.carbon.ml.core.impl.MLDatasetProcessor;
import org.wso2.carbon.ml.core.impl.MLModelHandler;
import org.wso2.carbon.ml.core.utils.MLUtils;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * This class is to build the Feed Forward Neural Network algorithm.
 */
public class FeedForwardNetwork {

    //global variables
    String mlDataset;
    double analysisFraction;
    String analysisResponceVariable ;
    int responseIndex;
    MLDatasetProcessor datasetProcessor = new MLDatasetProcessor() ;
    MLAnalysisHandler mlAnalysisHandler = new MLAnalysisHandler();
    MLModelHandler mlModelHandler=new MLModelHandler();
    PrivilegedCarbonContext carbonContext = PrivilegedCarbonContext.getThreadLocalCarbonContext();
    int tenantId = carbonContext.getTenantId();
    String userName = carbonContext.getUsername();

    /**
     * method to createFeedForwardNetwork.
     * @param seed
     * @param learningRate
     * @param analysisID
     * @param bachSize
     * @param backprop
     * @param hiddenList
     * @param inputLayerNodes
     * @param iterations
     * @param modelName
     * @param momentum
     * @param nepoches
     * @param noHiddenLayers
     * @param optimizationAlgorithms
     * @param outputList
     * @param pretrain
     * @param updater
     * @return an String object with evaluation result.
     */
    public String createFeedForwardNetwork(long seed, double learningRate, int bachSize, double nepoches, int iterations, String optimizationAlgorithms, String updater, double momentum, boolean pretrain, boolean backprop, int noHiddenLayers, int inputLayerNodes, String modelName, int analysisID, List<HiddenLayerDetails> hiddenList, List<OutputLayerDetails> outputList) throws IOException, InterruptedException {

        String evaluationDetails = null;
        int numLinesToSkip = 0;
        String delimiter = ",";
        mlDataset = getDatasetPath(modelName);
        analysisFraction = getAnalysisFraction(analysisID);
        analysisResponceVariable = getAnalysisResponseVariable(analysisID);
        responseIndex = getAnalysisResponseVariableIndex(analysisID);
        SplitTestAndTrain splitTestAndTrain;
        DataSet currentDataset;
        DataSet trainingset = null;
        DataSet testingset = null;
        INDArray features = null;
        INDArray labels = null;
        INDArray predicted = null;
        Random rnd = new Random();
        int labelIndex = 0;
        int numClasses = 0;
        int fraction = 0;

        //Initialize RecordReader
        RecordReader rr = new CSVRecordReader(numLinesToSkip,delimiter);
        //read the dataset
        rr.initialize(new FileSplit(new File(mlDataset)));
        labelIndex = responseIndex;
        numClasses = outputList.get(0).outputNodes;

        //Get the fraction to do the spliting data to training and testing
        FileReader fr = new FileReader(mlDataset);
        LineNumberReader lineNumberReader=new LineNumberReader(fr);
        //Get the total number of lines
        lineNumberReader.skip(Long.MAX_VALUE);
        int lines = lineNumberReader.getLineNumber();

        //handling multiplication of 0 error
        if(analysisFraction == 0){
            return null;
        }

        //Take floor value to set the numHold of training data
        fraction = ((int) Math.floor(lines * analysisFraction));

        org.nd4j.linalg.dataset.api.iterator.DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,lines,labelIndex,numClasses);

        //Create NeuralNetConfiguration object having basic settings.
        NeuralNetConfiguration.ListBuilder neuralNetConfiguration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(mapOptimizationAlgorithm(optimizationAlgorithms))
                .learningRate(learningRate)
                .updater(mapUpdater(updater))
                .momentum(momentum)
                .list(noHiddenLayers+1);

        //Add Hidden Layers to the network with unique settings
        for(int i = 0;i< noHiddenLayers;i++){
            int nInput = 0;
            if(i == 0)
                nInput=inputLayerNodes;
            else
                nInput=hiddenList.get(i-1).hiddenNodes;

            neuralNetConfiguration.layer(i,new DenseLayer.Builder().nIn(nInput)
                    .nOut(hiddenList.get(i).hiddenNodes)
                    .weightInit(mapWeightInit(hiddenList.get(i).weightInit))
                    .activation(hiddenList.get(i).activationAlgo)
                    .build());
        }

        //Add Output Layers to the network with unique settings
        neuralNetConfiguration.layer(noHiddenLayers, new OutputLayer.Builder(mapLossFunction(outputList.get(0).lossFunction))
                    .nIn(hiddenList.get(noHiddenLayers-1).hiddenNodes)
                    .nOut(outputList.get(0).outputNodes)
                    .weightInit(mapWeightInit(outputList.get(0).weightInit))
                    .activation(outputList.get(0).activationAlgo)
                    .build());

        //Create MultiLayerConfiguration network
        MultiLayerConfiguration conf = neuralNetConfiguration.pretrain(pretrain)
                .backprop(backprop).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));

        while (trainIter.hasNext()) {
            currentDataset = trainIter.next();
            splitTestAndTrain = currentDataset.splitTestAndTrain(fraction,rnd);
            trainingset = splitTestAndTrain.getTrain();
            testingset = splitTestAndTrain.getTest();
            features= testingset.getFeatureMatrix();
            labels = testingset.getLabels();
        }

        //Train the model with the training data
        for ( int n = 0; n < nepoches; n++) {
            model.fit( trainingset);
        }

        //Do the evaluations of the model including the Accuracy, F1 score etc.
        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(outputList.get(0).outputNodes);
        predicted = model.output(features,false);

        eval.eval(labels, predicted);

        evaluationDetails = "{\"Accuracy\":\""+eval.accuracy()+"\", \"Pecision\":\""+eval.precision()+"\",\"Recall\":\""+eval.recall()+"\",\"F1Score\":\""+eval.f1()+"\"}";
        return evaluationDetails;

    }

    /**
     * method to map user selected Optimazation Algorithm to OptimizationAlgorithm object.
     * @param optimizationAlgorithm
     * @return an OptimizationAlgorithm object.
     */
    OptimizationAlgorithm mapOptimizationAlgorithm(String optimizationAlgorithm){

        OptimizationAlgorithm optimizationAlgo = null;
        //selecting the relevent Optimization Algorithm
        if(optimizationAlgorithm.equals("Line_Gradient_Descent"))
            optimizationAlgo = OptimizationAlgorithm.LINE_GRADIENT_DESCENT;
        else if(optimizationAlgorithm.equals("Conjugate_Gradient"))
            optimizationAlgo = OptimizationAlgorithm.CONJUGATE_GRADIENT;
        else if(optimizationAlgorithm.equals("Hessian_Free"))
            optimizationAlgo = OptimizationAlgorithm.HESSIAN_FREE;
        else if(optimizationAlgorithm.equals("LBFGS"))
            optimizationAlgo = OptimizationAlgorithm.LBFGS;
        else if(optimizationAlgorithm.equals("Stochastic_Gradient_Descent"))
            optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;

        return optimizationAlgo;
    }

    /**
     * method to map user selected Updater Algorithm to Updater object.
     * @param updater
     * @return an Updater object .
     */
    Updater mapUpdater(String updater){

        Updater updaterAlgo = null;
        if(updater.equals("sgd"))
            updaterAlgo = Updater.SGD;
        else if (updater.equals("adam"))
            updaterAlgo = Updater.ADAM;
        else if (updater.equals("adadelta"))
            updaterAlgo = Updater.ADADELTA;
        else if (updater.equals("nesterovs"))
            updaterAlgo = Updater.NESTEROVS;
        else if (updater.equals("adagrad"))
            updaterAlgo = Updater.ADAGRAD;
        else if (updater.equals("rmsprop"))
            updaterAlgo = Updater.RMSPROP;
        else if (updater.equals("none"))
            updaterAlgo = Updater.NONE;
        else if (updater.equals("custom"))
            updaterAlgo = Updater.CUSTOM;
        return updaterAlgo;
    }

    /**
     * method to map user selected Loss Function Algorithm to LossFunction object.
     * @param lossFunction
     * @return an LossFunction object .
     */
    LossFunction mapLossFunction(String lossFunction){

        LossFunction lossfunctionAlgo = null;

        if(lossFunction.equals("mse"))
            lossfunctionAlgo = LossFunction.MSE;
        else if(lossFunction.equals("expll"))
            lossfunctionAlgo = LossFunction.EXPLL;
        else if(lossFunction.equals("xent"))
            lossfunctionAlgo = LossFunction.XENT;
        else if(lossFunction.equals("mcxent"))
            lossfunctionAlgo = LossFunction.MCXENT;
        else if(lossFunction.equals("rmsexent"))
            lossfunctionAlgo = LossFunction.RMSE_XENT;
        else if(lossFunction.equals("sqauredloss"))
            lossfunctionAlgo = LossFunction.SQUARED_LOSS;
        else if(lossFunction.equals("reconstructioncrossentropy"))
            lossfunctionAlgo = LossFunction.RECONSTRUCTION_CROSSENTROPY;
        else if(lossFunction.equals("negetiveloglilelihood"))
            lossfunctionAlgo = LossFunction.NEGATIVELOGLIKELIHOOD;
        else if(lossFunction.equals("custom"))
            lossfunctionAlgo = LossFunction.CUSTOM;
        return  lossfunctionAlgo;
    }

    /**
     * method to map user selected WeightInit Algorithm to WeightInit object.
     * @param weightinit
     * @return an WeightInit object .
     */
    WeightInit mapWeightInit(String weightinit){

        WeightInit weightInitAlgo = null;
        if(weightinit.equals("Distribution"))
            weightInitAlgo = WeightInit.DISTRIBUTION;
        else if(weightinit.equals("Normalized"))
            weightInitAlgo = WeightInit.NORMALIZED;
        else if(weightinit.equals("Size"))
            weightInitAlgo = WeightInit.SIZE;
        else if(weightinit.equals("Uniform"))
            weightInitAlgo = WeightInit.UNIFORM;
        else if(weightinit.equals("Vi"))
            weightInitAlgo = WeightInit.VI;
        else if(weightinit.equals("Zero"))
            weightInitAlgo = WeightInit.ZERO;
        else if(weightinit.equals("Xavier"))
            weightInitAlgo = WeightInit.XAVIER;
        else if(weightinit.equals("RELU"))
            weightInitAlgo = WeightInit.RELU;
        else if(weightinit.equals("Normalized"))
            weightInitAlgo = WeightInit.NORMALIZED;

        return weightInitAlgo;
    }

    /**
     * method to analysis fraction from mlAnalysisHandler.
     * @param analysisId
     * @return analysis fraction .
     */
    double getAnalysisFraction(long analysisId) {
        try {
            double trainDataFraction = mlAnalysisHandler.getTrainDataFraction(analysisId);
            return  trainDataFraction;

        } catch (MLAnalysisHandlerException e) {
            String msg = MLUtils.getErrorMsg(String.format(
                  "Error occurred while retrieving train data fraction for the analysis [id] %s of tenant [id] %s and [user] %s .",
                   analysisId, tenantId, userName), e);
            return 0.0;
        }
    }

    /**
     * method to get analysis Responsible Variable from mlAnalysisHandler.
     * @param analysisId
     * @return Response Variable .
     */
    String getAnalysisResponseVariable(long analysisId) {
        try {
            String responseVariable = mlAnalysisHandler.getResponseVariable(analysisId);
            return  responseVariable;
        } catch (MLAnalysisHandlerException e) {
            String msg = MLUtils.getErrorMsg(String.format(
                   "Error occurred while retrieving train data fraction for the analysis [id] %s of tenant [id] %s and [user] %s .",
                    analysisId, tenantId, userName), e);
            return null;
        }
    }

    /**
     * method to get analysis Responsible Variable index.
     * @param analysisId
     * @return Response Variable index.
     */
    int getAnalysisResponseVariableIndex(long analysisId) {
        try {
            List<String> features = mlAnalysisHandler.getFeatureNames(Long.toString(analysisId));
            int index= features.indexOf(analysisResponceVariable);
            return  index;
        } catch (MLAnalysisHandlerException e) {
            String msg = MLUtils.getErrorMsg(String.format(
                   "Error occurred while retrieving index of the current response feature for the analysis [id] %s of tenant [id] %s and [user] %s .",
                   analysisId, tenantId, userName), e);
            return -1;
        }
    }

    /**
     * method to get dataset version path.
     * @param modelName
     * @return DAtaset version stored path-target path.
     */
    String getDatasetPath(String modelName) {
        PrivilegedCarbonContext carbonContext = PrivilegedCarbonContext.getThreadLocalCarbonContext();

        int tenantId = carbonContext.getTenantId();
        String userName = carbonContext.getUsername();
        try {
            MLModelData model = mlModelHandler.getModel(tenantId, userName, modelName);
            long versionSetId = model.getVersionSetId();
            String path = datasetProcessor.getVersionset(tenantId, userName,versionSetId).getTargetPath();

            if (model == null) {
                return null;
            }
            return path;

        } catch (MLModelHandlerException e) {
            String msg = MLUtils.getErrorMsg(String.format(
                    "Error occurred while retrieving a model [name] %s of tenant [id] %s and [user] %s .", modelName,
                    tenantId, userName), e);
                return null;
        } catch (MLDataProcessingException e) {
            e.printStackTrace();
            return null;
        }
    }
}
