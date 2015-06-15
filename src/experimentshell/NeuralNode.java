/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experimentshell;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 *
 * @author mormon
 */
public class NeuralNode {
    public int m_numInputs; // for the inputs

    // the weights for each input
    public List<Double> m_weights = new ArrayList();

    public NeuralNode(int numInputs) {
        // add additional neuron for bias
        m_numInputs = numInputs + 1;

        Random rand = new Random();
        double randomNum;
        // randomize the weights and add them to the list
        for (int i = 0; i < m_numInputs; i++) {
            randomNum = rand.nextDouble() * 2 - 1;
            m_weights.add(randomNum);
        }
    }
}