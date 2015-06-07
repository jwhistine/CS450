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
public class NeuralLayer {
    public int m_numNeurons;
   
    public List<Double> listActivations;
    public List<Double> listError;
    
    List<NeuralNode> m_neurons;
	
    public NeuralLayer(int numNeurons, int numInputPerNeuron) {
        m_neurons = new ArrayList();
        m_numNeurons = numNeurons;
        
        for (int i = 0; i < numNeurons; i++) {
            NeuralNode n = new NeuralNode(numInputPerNeuron);
            m_neurons.add(n);
        }
    }
    
    public int getNumNeurons() {
        return m_numNeurons;
    }
}