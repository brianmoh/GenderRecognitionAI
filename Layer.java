// Layer of Sigmoid Neurons

import java.util.ArrayList;

public class Layer {
    private ArrayList<SigmoidNeuron> nodes = new ArrayList<>();             // List of nodes in this layer
    //private final int prevSize;                                             // Number of nodes in the previous layer
    
    //@param   size        Number of neurons in this layer
    //@param   vectorSize  Size of input vectors
    public Layer (int size, int vectorSize){
        for(int i=0; i<size; i++)
            nodes.add(new SigmoidNeuron(vectorSize));
        //prevSize = vectorSize;
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
    
    //Gets the 
}
