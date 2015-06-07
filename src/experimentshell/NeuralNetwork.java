/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experimentshell;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author mormon
 */
public class NeuralNetwork {
    int m_numInputs;
    int m_numOutputs;
    int m_numHiddenLayers;
    int m_neuronsPerHiddenLayer;
    
    private double bias = -1.0;
    private double response = 1.0;
    private double learningRate = 0.25;
    
    List<NeuralLayer> m_layers = new ArrayList(); // stores each layer of nodes

    NeuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int nueronsPerHiddenLayer) {
        m_numInputs = numInputs;
        m_numOutputs = numOutputs;
        m_numHiddenLayers = numHiddenLayers;
        m_neuronsPerHiddenLayer = nueronsPerHiddenLayer;
        
        if (m_numHiddenLayers > 0) {
            // create the hidden layer
            NeuralLayer nLayer = new NeuralLayer(m_neuronsPerHiddenLayer, m_numInputs);
            m_layers.add(nLayer);
            
            for (int i = 0; i < m_numHiddenLayers - 1; i++) {
                NeuralLayer hiddenLayer = new NeuralLayer(m_neuronsPerHiddenLayer, m_neuronsPerHiddenLayer);
                m_layers.add(hiddenLayer);
            }
            
            // create output layer
            NeuralLayer outputLayer = new NeuralLayer(m_numOutputs, m_neuronsPerHiddenLayer);
            m_layers.add(outputLayer);
        }
        else {
            // create output layer
            NeuralLayer outputLayer2 = new NeuralLayer(m_numOutputs, m_numInputs);
            m_layers.add(outputLayer2);
        }
    }
 	
    // calculates the outputs from a set of inputs
    public List<Double> feedForward(List<Double> inputs) {
        List<Double> outputs = new ArrayList(); // stores the result
	int weightIndex;

	// check for correct number of outputs
	if (inputs.size() != m_numInputs)
            return outputs; // return empty list if incorrect

	// iterate through....
	for (int i = 0; i < m_numHiddenLayers + 1; i++) {
            if (i > 0) {
		inputs = new ArrayList(outputs);
            }
            outputs.clear();
            
            weightIndex = 0;

            // sum the inputs and the weights and add them to the sigmoid
            // function to be calculated.
            for (int j = 0; j < m_layers.get(i).m_numNeurons; j++) {
		double weightedSum = 0.0;
		
                // number of inputs for the layer
                int numInputs = m_layers.get(i).m_neurons.get(j).m_numInputs;

		// for each weight given for calculation
		for (int k = 0; k < numInputs - 1; k++) {
        	    // sum the weights
		    weightedSum += m_layers.get(i).m_neurons.get(j).m_weight.get(k)
                            * inputs.get(weightIndex++);
		}
                
                // add the bias to make it work
                weightedSum += m_layers.get(i).m_neurons.get(j).m_weight.get(numInputs - 1) * bias;
                
                // calculate the "h" for the layer (output)
                outputs.add(sigmoid(weightedSum, response));
                
                weightIndex = 0;
            }
            m_layers.get(i).listActivations = new ArrayList(outputs);
	}
        return outputs;
    }

    // response curve (s-figure)
    public double sigmoid(double netInput, double response) {
        // from the book
        return (1.0 / (1.0 + Math.exp(-netInput / response)));
    }
    
    void backPropagate(double classIndex) {
        double error, target, activation, weightValue, weightedSum;
        weightedSum = 0;
        List<Double> listError = new ArrayList();
        
        // output layer calculation
        for (int i = 0; i < m_layers.get(m_numHiddenLayers).m_numNeurons; i++) {
            if (i == classIndex)
                target = 1.0;
            else
                target = 0.0;
            
            activation = m_layers.get(m_numHiddenLayers).listActivations.get(i);
            error = activation * (1 - activation) * (activation - target);
            listError.add(error);
        }
        
        m_layers.get(m_numHiddenLayers).listError = new ArrayList(listError);
        
        // calculate the hidden layer
        if (m_numHiddenLayers > 0) {
            listError.clear();
            for (int i = m_numHiddenLayers; i > 0; i--) {
                // each node in layer
                for (int j = 0; i < m_layers.get(i - 1).m_numNeurons; j++) {
                    activation = m_layers.get(i - 1).listActivations.get(j);
                    for (int k = 0; k < m_layers.get(i).m_numNeurons; k++) {
                        weightedSum += m_layers.get(i).listError.get(k) * m_layers.get(i).m_neurons.get(k).m_weight.get(j);
                    }
                    error = activation * (1 - activation) * weightedSum;
                    listError.add(error);
                }
                m_layers.get(i - 1).listError = new ArrayList(listError);
            }
        }
        
        // feedforward update
        for (int i = 0; i < m_numHiddenLayers + 1; i++) {
            for (int j = 0; j < m_layers.get(i).m_numNeurons; j++) {
                int numInputs = m_layers.get(i).m_neurons.get(j).m_numInputs;
                for (int k = 0; k < numInputs; k++) {
                    weightValue = m_layers.get(i).m_neurons.get(j).m_weight.get(k);
                    weightValue -= learningRate * m_layers.get(i).listError.get(j)
                            * m_layers.get(i).listActivations.get(j);
                    m_layers.get(i).m_neurons.get(j).m_weight.set(k, weightValue);
                }
            }
        }
    }
}