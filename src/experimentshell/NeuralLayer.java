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
    // the number of neurons in this layer
    public int mNumNeurons;

    // to store the output/activation
    public List<Double> mListActivations;

    public List<Double> mListError;

    // the layer of neurons
    public List<NeuralNode> mListNeurons;

    public NeuralLayer(int numNeurons, int numInputPerNueron) {
        mListNeurons = new ArrayList();
        mNumNeurons = numNeurons;
        for (int i = 0; i < mNumNeurons; i++) {
            NeuralNode n = new NeuralNode(numInputPerNueron);
            mListNeurons.add(n);
        }

    }

    public int getmNumNeurons() {
        return mNumNeurons;
    }
}