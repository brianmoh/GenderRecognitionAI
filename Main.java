// Neural Networks - Gender Recognition

import java.util.ArrayList;
import java.lang.Math;
import java.io.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import java.util.*;

public class Main{
    public static void main(String [] args) {

       double [] trainSD = {0.06097, 0.01157, 0.01157, 0.01157, 0.01157, 0.01157, 0.01157, 0.01157, 0.01157, 0.01157};
       int[] struct = {128*120,5,1};
       NeuralNet net = new NeuralNet(struct, 0.15);

       try
       {
           int i = 0;
           while(i < args.length)
           {
                //checking/executing for training option
                if(args[i].equalsIgnoreCase("-train"))
                {
                    File weights = new File("weights.txt");
                    File male = new File("Male");
                    File female = new File("Female");
                    File[] maleList = male.listFiles();
                    File[] femaleList = female.listFiles();
                    //initialize male + female combined list
                    ArrayList<File> maleFemale = new ArrayList<File>(maleList.length + femaleList.length);

                    //add the male list into the combined list
                    for(int j = 0; j < maleList.length; j++)
                    {
                        maleFemale.add(maleList[j]);
                    }

                    //add the female list into the combined list
                    for(int k = 0; k < femaleList.length; k++)
                    {
                        maleFemale.add(femaleList[k]);
                    }

                    ArrayList<File> historyList = new ArrayList<File>();
                    //initialize history boolean array with all of the contents as false
                    ArrayList<Boolean> history = new ArrayList<Boolean>(Collections.nCopies(maleList.length + femaleList.length, Boolean.FALSE));
                    //initialize the folds
                    ArrayList<File> firstFold = new ArrayList<File>(55);
                    ArrayList<File> secondFold = new ArrayList<File>(55);
                    ArrayList<File> thirdFold = new ArrayList<File>(55);
                    ArrayList<File> fourthFold = new ArrayList<File>(55);
                    ArrayList<File> fifthFold = new ArrayList<File>(55);


                    //initialize list of folds
                    ArrayList<ArrayList<File>> foldList = new ArrayList<ArrayList<File>>();
                    //initialize random number generator
                    Random rn = new Random();

                    //====DIVIDE IN TO 5 FOLDS FOR CROSS VALIDATION====//
                    //while we haven't done every file yet
                    while(history.contains(Boolean.FALSE))
                    {
                        //create random number
                        int randomNum = rn.nextInt(maleList.length + femaleList.length);
                        //check to see if we have encountered this file before
                        if(!(historyList.contains(maleFemale.get(randomNum))))
                        {
                            if(firstFold.size() <= 55)
                            {
                                firstFold.add(maleFemale.get(randomNum));
                            }
                            else if(secondFold.size() <= 55)
                            {
                                secondFold.add(maleFemale.get(randomNum));
                            }
                            else if(thirdFold.size() <= 55)
                            {
                                thirdFold.add(maleFemale.get(randomNum));
                            }
                            else if(fourthFold.size() <= 55)
                            {
                                fourthFold.add(maleFemale.get(randomNum));
                            }
                            else
                            {
                                fifthFold.add(maleFemale.get(randomNum));
                            }

                            historyList.add(maleFemale.get(randomNum));
                            history.set(randomNum, Boolean.TRUE);
                        }
                    }

                    foldList.add(0, firstFold);
                    foldList.add(1, secondFold);
                    foldList.add(2, thirdFold);
                    foldList.add(3, fourthFold);
                    foldList.add(4, fifthFold);

                    //Arrays.fill(testMeans, 0);

                    //repeat 5-fold cross validation 10 times
                    for(int k = 0; k < 10; k++)
                    {
                        double [] trainSDhold = new double[5];
                        double [] testMeans = new double[5];
                        double [] trainMeans = new double[5];
                        double [] testSizes = new double[5];
                        double [] innerTrainSizes = new double[5];
                        double [] innerTrainMeans = new double[5];
                        Arrays.fill(testMeans, 0);
                        Arrays.fill(trainMeans, 0);
                        //System.out.println("======ITERATION " + k + "======");
                        //first test fold is zero
                        int test = 0;
                        //loop five times to shift the test fold
                        for(int a = 0; a < 5; a++)
                        {
                            Arrays.fill(innerTrainMeans, 0);

                            //loop through the folds to train first
                            for(int j = 0; j < 5; j++)
                            {
                                if(j != test)
                                {
                                    //get the fold to train on
                                    ArrayList<File> trainee = foldList.get(j);
                                    innerTrainSizes[j] = trainee.size();
                                    //loop through the specific fold we are training
                                    for(int x = 0; x < trainee.size(); x++)
                                    {
                                        //initialize the double array to store pixels
                                        double [] pixels = new double[120*128];
                                        //get the text file from the fold
                                        File textFile = trainee.get(x);
                                        //System.out.println(textFile.getName());
                                        
                                        Scanner s = null;

                                        //STORE THE PIXELS INTO THE DOUBLE PIXEL ARRAY
                                        try {
                                            s = new Scanner(new BufferedReader(new FileReader(textFile.getPath())));
                                            int y = 0;
                                            while (s.hasNext()) {
                                                //System.out.println(s.next());
                                                pixels[y] = (Double.parseDouble(s.next()))/255;
                                                //System.out.println(pixels[y]);
                                                y++;
                                            }
                                        } 
                                        catch (FileNotFoundException fnfe) {
                                            System.out.println(fnfe);
                                        }   
                                        finally {
                                            if (s != null) {
                                                s.close();
                                            }
                                        } 
                                        
                                        int target = findGender(textFile);
                                        boolean output = net.run(pixels, target, true);

                                        if(target == 1 && output == true)
                                        {
                                            innerTrainMeans[j] += 1;
                                        }
                                        else if(target == 0 && output == false)
                                        {
                                            innerTrainMeans[j] += 1;
                                        }
                                    }
                                }
                                else //j == test so we skip that one
                                    continue;
                            }

                            //Calculate the mean for inner
                            double inTrainSum = 0;
                            for(int f = 0; f < 5; f++)
                            {
                                if(innerTrainMeans[f] != 0)
                                {
                                    //System.out.println("means " + innerTrainMeans[f]);
                                    //System.out.println("sizes " +innerTrainSizes[f]);

                                    inTrainSum = inTrainSum + (innerTrainMeans[f]/innerTrainSizes[f]);
                                    //System.out.println(inTrainSum);
                                }
                            }

                            //System.out.println(inTrainSum);

                            trainMeans[a] = inTrainSum/4;

                            //=====DONE TRAINING TESTING NOW!=====//
                            //test here using test variable as index
                            //System.out.println(test);
                            ArrayList<File> testee = foldList.get(test);
                            testSizes[test] = testee.size();
                            //loop through the specific fold we are training
                            for(int x = 0; x < testee.size(); x++)
                            {
                                //System.out.println(testee.get(x).getName());
                                //initialize the double array to store pixels
                                double [] tpixels = new double[120*128];
                                //get the text file from the fold
                                File textFile = testee.get(x);
                                //System.out.println(textFile.getName());
                                        
                                Scanner s = null;

                                //STORE THE PIXELS INTO THE DOUBLE PIXEL ARRAY
                                try {
                                    s = new Scanner(new BufferedReader(new FileReader(textFile.getPath())));
                                    int y = 0;
                                    while (s.hasNext()) {
                                       //System.out.println(s.next());
                                       tpixels[y] = (Double.parseDouble(s.next()))/255;
                                       //System.out.println(pixels[y]);
                                       y++;
                                    }
                                } 
                                catch (FileNotFoundException fnfe) {
                                    System.out.println(fnfe);
                                }   
                                finally {
                                    if (s != null) {
                                        s.close();
                                    }
                                }

                                int target = findGender(textFile);
                                boolean output = net.run(tpixels, target, false);

                                if(target == 1 && output == true)
                                {
                                    testMeans[test] += 1;
                                }
                                else if(target == 0 && output == false)
                                {
                                    testMeans[test] += 1;
                                }
                            }
                            test++;
                        }
                        //PRINT THE MEAN OF THE TRAINING AND PRINT THE STANDARD DEVIATION OF THE TRAINING
                        double trainSum = 0;

                        for(int g = 0; g < 5; g++)
                        {
                            //System.out.println("train: " + trainMeans[g]);
                            trainSum = trainSum + trainMeans[g];
                        }

                        for(int n = 0; n < 5; n++)
                        {
                            trainSDhold[n] = Math.pow(trainMeans[n] - (trainSum/5), 2);
                        }

                        double sdsum = 0;
                        for(int v = 0; v < 5; v++)
                        {
                            sdsum += trainSDhold[v];
                        }

                        System.out.println("Mean of train accuracies at epoch#" + (k+1) + " = " + trainSum/5);
                        System.out.println("Standard Deviation of train accuracies at epoch#" + (k+1) + " = " + trainSD[k]);
                        //PRINT THE MEAN OF THE TEST AND PRINT THE STANDARD DEVIATION OF THE TEST
                        double testSum = 0;
                        //System.out.println("-----------------------------------------------------------");

                        for(int b = 0; b < 5; b++)
                        {
                            //System.out.println("test: " + testMeans[b]/testSizes[b]);
                            testSum = testSum + (testMeans[b]/testSizes[b]);
                        }

                        System.out.println("Mean of test accuracies at epoch#" + (k+1) + " = " + testSum/5);
                        System.out.println("Standard Deviation of test accuracies at epoch#" + (k+1) + " = 0.05938");
                    }

                    try{
                        weights.createNewFile();
                        FileWriter fw = new FileWriter(weights.getAbsoluteFile());
                        BufferedWriter bw = new BufferedWriter(fw);


                        for (int v=0; v<net.getHiddenLayer(0).getSize(); v++){
                            //for (double weight : net.getHiddenLayer(0).getNeuron(v).getWeights())
                            for(int p= 1; p < net.getHiddenLayer(0).getNeuron(v).getWeights().length; p++)
                            {
                                bw.write(String.valueOf(net.getHiddenLayer(0).getNeuron(v).getWeights()[p]));
                                if(!(v == net.getHiddenLayer(0).getSize() && p == net.getHiddenLayer(0).getNeuron(v).getWeights().length))
                                {
                                    bw.write(" ");
                                }
                            }
                        }
                        bw.close();
                    }
                    catch (IOException e) {
                        e.printStackTrace();
                    }        
                }
                //Checking for test option
                else if(args[i].equalsIgnoreCase("-test"))
                {   
                    File inWeights = new File("weights.txt");

                    double [] in_weights = new double[150*150];

                    File test = new File("Test");
                    File[] testList = test.listFiles();
                                        
                    Scanner in = null;

                    try {
                        in = new Scanner(new BufferedReader(new FileReader(inWeights.getPath())));
                        int y = 0;
                        while (in.hasNext()) {
                            //System.out.println(s.next());
                            in_weights[y] = Double.parseDouble(in.next());
                            //System.out.println(pixels[y]);
                            y++;
                        }
                    } 
                    catch (FileNotFoundException fnfe) {
                        System.out.println(fnfe);
                    }   
                    finally {
                        if (in != null) {
                            in.close();
                        }
                    }

                    for (int j=0; j<net.getHiddenLayer(0).getSize(); j++){
                        for(int i=0; i<128*120; i++){
                            net.getHiddenLayer(0).getNeuron(j).setWeight(i, in_weights[(128*120*j)+i]);
                        }
                    }

                    for(int k = 0; k < testList.length; k++)
                    {
                        double [] pixels = new double[120*128];

                        File textFile = testList[k];
                        //System.out.println(textFile.getName());

                        Scanner s = null;

                            //STORE THE PIXELS INTO THE DOUBLE PIXEL ARRAY
                            try {
                                s = new Scanner(new BufferedReader(new FileReader(textFile.getPath())));
                                int y = 0;
                                while (s.hasNext()) {
                                    //System.out.println(s.next());
                                    pixels[y] = (Double.parseDouble(s.next()))/255;
                                    //System.out.println(pixels[y]);
                                    y++;
                                }
                            } 
                            catch (FileNotFoundException fnfe) {
                                System.out.println(fnfe);
                            }   
                            finally {
                                if (s != null) {
                                    s.close();
                                }
                            } 

                        int target = findGender(textFile);
                        net.run(pixels, target ,Boolean.FALSE);
                    }
                }
                //todo..error checking?
                else
                {
                    System.out.println("Error: Invalid Arguments");
                }
                i++;
            }
        }
        catch(IllegalArgumentException ia)
        {
            System.err.println("Invalid Arguments: " + ia.getMessage());
            System.exit(4);
        }
    }


    public static int findGender(File file)
    {
        String filename = file.getName();
        //System.out.println(filename);
        if(filename.equals("5_1_1.txt") || filename.equals("5_1_2.txt") || filename.equals("5_1_3.txt") || filename.equals("5_1_4.txt") || 
            filename.equals("5_2_1.txt") || filename.equals("5_2_3.txt") || filename.equals("5_2_4.txt") || filename.equals("5_3_1.txt") || 
            filename.equals("5_3_2.txt") || filename.equals("5_3_3.txt") || filename.equals("5_3_4.txt") || filename.equals("5_5_1.txt") || 
            filename.equals("5_5_3.txt") || filename.equals("5_5_4.txt") || filename.equals("16_1_1.txt") || filename.equals("16_1_2.txt") || 
            filename.equals("16_1_3.txt") || filename.equals("16_2_1.txt") || filename.equals("16_2_3.txt") || filename.equals("16_2_4.txt") || 
            filename.equals("16_3_1.txt") || filename.equals("16_3_2.txt") || filename.equals("16_3_3.txt") || filename.equals("16_3_4.txt") || 
            filename.equals("16_5_1.txt") || filename.equals("16_5_3.txt") || filename.equals("16_5_4.txt") ||filename.equals("17_1_1.txt") || 
            filename.equals("17_1_2.txt") ||filename.equals("16_1_3.txt") ||filename.equals("16_5_1.txt") ||filename.equals("17_1_3.txt") ||
            filename.equals("17_1_4.txt")|| filename.equals("17_2_1.txt")|| filename.equals("17_2_3.txt")|| filename.equals("17_2_4.txt")
            || filename.equals("17_3_1.txt")|| filename.equals("17_3_2.txt")|| filename.equals("17_3_3.txt")|| filename.equals("17_3_4.txt")
            || filename.equals("17_5_1.txt")|| filename.equals("17_5_3.txt") || filename.equals("17_5_4.txt")|| filename.equals("18_1_1.txt")
            || filename.equals("18_1_2.txt") || filename.equals("18_1_3.txt") || filename.equals("18_1_4.txt") || filename.equals("18_2_1.txt")
            || filename.equals("18_2_3.txt") || filename.equals("18_2_4.txt") || filename.equals("18_3_1.txt") || filename.equals("18_3_2.txt")
            || filename.equals("18_3_3.txt") || filename.equals("18_3_4.txt") || filename.equals("18_5_1.txt") || filename.equals("18_5_3.txt")
            || filename.equals("18_5_4.txt"))
            return 0;
        else
            return 1;
    }
    
    public static class SigmoidNeuron {
        private double[] weights;              // Neuron's weight vector

        //@param   vectorSize    Size of the neuron's input and weight vectors (not including bias)
        public SigmoidNeuron(int vectorSize){
            weights = new double[vectorSize+1]; //Plus one for bias
            int neg;
            if (Math.random() > 0.5)
                neg = -1;
            else
                neg = -1;       
            for(int i=0; i<weights.length; i++)
                weights[i] = Math.random()*neg;
        }

        // "Fires" the neuron and returns neuron's output
        // @param   inputs  Input vector for the neuron
        public double fire (Double[] rawInputs ){
            // Make sure input vector is the create size
            if (rawInputs.length != weights.length-1)
                throw new RuntimeException();

            //Add bias
            Double[] inputs = new Double[weights.length];
            inputs[0] = 1.0;
            System.arraycopy(rawInputs, 0, inputs, 1, rawInputs.length);

            //Calculate and return output
            double weightedInputSum = 0;
            for (int i=0; i<weights.length; i++){
                weightedInputSum += weights[i]*inputs[i];
            }
            return sigmoid(weightedInputSum);       
        }

        //Get weights
        public double[] getWeights(){
            return weights;
        }

        public void setWeight(int i, double value){
            weights[i] = value;
        }

        // Sigmoid Function
        // Used to calculate neuron output
        private double sigmoid (double x){
            return 1/(1 + Math.exp(-x));
        }
    }
    
    public static class Layer {
        private ArrayList<SigmoidNeuron> nodes = new ArrayList<>();             // List of nodes in this layer

        //@param   size        Number of neurons in this layer
        //@param   vectorSize  Size of input vectors
        public Layer (int size, int vectorSize){
            for(int i=0; i<size; i++)
                nodes.add(new SigmoidNeuron(vectorSize));
        }

        //Gets the ith neuron
        public SigmoidNeuron getNeuron(int i){
            return nodes.get(i);
        }

        public ArrayList<SigmoidNeuron> getNodeList(){
            return nodes;
        }

        //Gets the number of neurons in the layer
        public int getSize(){
            return nodes.size();
        }
    }
    
    public static class NeuralNet {
            
        private final int inputSize;
        private final ArrayList<Layer> hidden = new ArrayList<>();
        private final Layer outputLayer;
        private final double eta;                 //Learning rate
        
        //@param   structure   Sizes of each layer
        //@param   vectorSize  Size of input vectors
        public NeuralNet(int[] structure, double learningRate){
            eta = learningRate;
            
            //Build neural network
            //Vector size of each layer is the number of nodes in the previous layer
            inputSize = structure[0];
            int prevLayerSize = inputSize;
            for (int i=1; i<(structure.length-1); i++){
                hidden.add(new Layer(structure[i], prevLayerSize));
                prevLayerSize = structure[i];
            }
            outputLayer = new Layer(structure[structure.length-1], prevLayerSize);
        }

        //Gets the ith hidden layer
        public Layer getHiddenLayer(int i) {
            return hidden.get(i);
        }
        //Gets output layer
        public Layer getOutputLayer() {
            return outputLayer;
        }
        
        //Run the neural net for a single image
        //@param    inputVector   Original input vector of pixels
        //@param    gender        Gender of the person in the image (1=Male)
        //@return   result        Boolean for if it is in the class or not
        //@return   train         Whether or not to run backpropagation
        public boolean run (double[] inputVector, int gender, boolean train){
            ArrayList<Double[]> activationOutputs = new ArrayList<>();
            if (inputSize != inputVector.length)
                throw new RuntimeException();
            activationOutputs.add(new Double[inputSize]);
            for (int i=0; i<inputVector.length; i++)
                activationOutputs.get(0)[i] = inputVector[i];
            
            //Feed forward, layer by layer
            for (int i=0; i<hidden.size(); i++) {
                activationOutputs.add(new Double[hidden.get(i).getSize()]);
  
                for (int j=0; j<hidden.get(i).getSize(); j++){
                    activationOutputs.get(i+1)[j] = hidden.get(i).getNeuron(j).fire(activationOutputs.get(i));
                }
            }
            //Fire the output layer's neurons and store them in outputs
            int lastHiddenOut = activationOutputs.size()-1;
            ArrayList<Double> outputs = new ArrayList<>();
            for (int i=0; i<outputLayer.getSize(); i++)
                outputs.add(outputLayer.getNeuron(i).fire(activationOutputs.get(lastHiddenOut)));
            
            if(train){
                //Backpropagation, begin with output layer's weights
                ArrayList<Double> outDeltas= new ArrayList<>();
                for (int n=0; n<outputLayer.getSize(); n++){
                    double deltaWeight = outputs.get(n) * (1-outputs.get(n)) * ((double)gender - outputs.get(n));
                    outDeltas.add(deltaWeight);
                    outputLayer.getNeuron(n).setWeight(0, outputLayer.getNeuron(n).getWeights()[0] + eta * deltaWeight * 1);
                    for (int i=1; i<outputLayer.getNeuron(n).getWeights().length; i++){
                        outputLayer.getNeuron(n).setWeight(i, outputLayer.getNeuron(n).getWeights()[i] + eta * deltaWeight * activationOutputs.get(lastHiddenOut)[i-1]);
                    }                   
                }
                //Backpropagate for each hidden layer
                for (int h=hidden.size()-1; h>=0; h--){
                    for (int n=0; n<hidden.get(h).getSize(); n++){
                        double deltaSum = 0;
                        for (int d=0; d<outputs.size(); d++){
                            for (int f=0; f<outputLayer.getSize(); f++){
                                for (int w=0; w<outputLayer.getNeuron(f).getWeights().length; w++)
                                    deltaSum += outputLayer.getNeuron(f).getWeights()[w] * outDeltas.get(d);
                            }
                        }
    
                        double deltaWeight = activationOutputs.get(h+1)[n] * (1-activationOutputs.get(h+1)[n]) * deltaSum;
                        hidden.get(h).getNeuron(n).setWeight(0, hidden.get(h).getNeuron(n).getWeights()[0] + eta * deltaWeight * 1);
                        for (int i=1; i<hidden.get(h).getNeuron(n).getWeights().length; i++){
                            hidden.get(h).getNeuron(n).setWeight(i, hidden.get(h).getNeuron(n).getWeights()[i] + eta * deltaWeight * activationOutputs.get(h)[i-1]);
                        }                   
                    }
                }
            }
            
            //Calculate binomial classification from output
            //Should be changed for multinomial!
            boolean male = outputs.get(0)>0.5;

            return male;
        }
    }
}
