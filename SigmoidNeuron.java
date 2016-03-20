// Sigmoid Neuron

import java.lang.Math;

public class SigmoidNeuron {
    private double[] weights;              // Neuron's weight vector
    
    //@param   vectorSize    Size of the neuron's input and weight vectors (not including bias)
    public SigmoidNeuron(int vectorSize){
        weights = new double[vectorSize+1]; //Plus one for bias
        int neg;
        if(Math.random()>0.5)
            neg = 1;
        else
            neg = -1;

        for(int i=0; i<weights.length; i++)
            weights[i] = Math.random() * neg;
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
