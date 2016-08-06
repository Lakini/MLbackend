package org.wso2.carbon.ml.rest.api.neuralNetworks;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
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
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.springframework.core.io.ClassPathResource;
import java.io.IOException;
import java.util.Collections;
import java.util.List;

public class FeedForwardNetwork {

    /**
     * method to createFeedForwardNetwork.
     * @return an String object with evaluation result.
     */
    public String createFeedForwardNetwork(long seed, double learningRate, int bachSize, double nepoches, int iterations, String optimizationAlgorithms, String updater, double momentum, boolean pretrain, boolean backprop, int noHiddenLayers, int inputLayerNodes, List<HiddenLayerDetails> hiddenList, List<OutputLayerDetails> outputList) throws IOException, InterruptedException {

        int numLinesToSkip = 0;
        String delimiter = ",";

        //Read the dataset for Training and Testing
        RecordReader rr = new CSVRecordReader(numLinesToSkip,delimiter);
        rr.initialize(new FileSplit(new ClassPathResource("iris_data_training.txt").getFile()));

        int labelIndex = 4;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        org.nd4j.linalg.dataset.api.iterator.DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,84,labelIndex,numClasses);
        RecordReader rrTest = new CSVRecordReader(numLinesToSkip,delimiter);
        rrTest.initialize(new FileSplit(new ClassPathResource("iris_data_testing.txt").getFile()));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,65,labelIndex,numClasses);

        //Create NeuralNetConfiguration object having basic settings.
        NeuralNetConfiguration.ListBuilder neuralNetConfiguration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(mapOptimizationAlgorithm(optimizationAlgorithms))
                .learningRate(learningRate)
                .updater(mapUpdater(updater))
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

        //Train the model with the training data
        for ( int n = 0; n < nepoches; n++) {
            model.fit( trainIter );
        }

        //Do the evaluations of the model including the Accuracy, F1 score etc.
        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(outputList.get(0).outputNodes);
        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features,false);

            eval.eval(lables, predicted);
        }

        return eval.stats();
    }

    /**
     * method to map user selected Optimazation Algorithm to OptimizationAlgorithm object.
     * @param optimizationAlgorithm which is a String value
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
     * @param updater which is a String value
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
     * @param lossFunction which is a String value
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
     * @param weightinit which is a String value
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

}
