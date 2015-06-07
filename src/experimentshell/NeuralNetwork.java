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
    private int m_numInputs;

    private int m_numOutputs;

    private int m_numHiddenLayers;

    private int m_neuronsPerHiddenLayer;

    // storage for each layer of neurons including the output layer
    List<NeuralLayer> mListLayers = new ArrayList();

    private double mBias = 1.0;
    private double mResponse = 1.0;
    private double mLearningRate = 0.25;

    public NeuralNetwork(int pNumInputs, int pNumOutputs,
            int pNumHiddenLayers, int pNeuronsPerHiddenLayer) {
        m_numInputs = pNumInputs;
        m_numOutputs = pNumOutputs;
        m_numHiddenLayers = pNumHiddenLayers;
        m_neuronsPerHiddenLayer = pNeuronsPerHiddenLayer;

        // the weights are all initially set to random values -1 < w < 1
        // create the layers of the network
        if (m_numHiddenLayers > 0) {

            // create first hidden layer
            NeuralLayer nLayer = new NeuralLayer(m_neuronsPerHiddenLayer, m_numInputs);
            mListLayers.add(nLayer);

            // create additional hidden layers
            for (int i = 0; i < m_numHiddenLayers - 1; i++) {
                NeuralLayer hiddenLayer = new NeuralLayer(m_neuronsPerHiddenLayer, m_neuronsPerHiddenLayer);
                mListLayers.add(hiddenLayer);
            }

            // create output layer
            NeuralLayer outputlayer = new NeuralLayer(m_numOutputs, m_neuronsPerHiddenLayer);
            mListLayers.add(outputlayer);
        } else {
            // create output layer
            NeuralLayer outLayer2 = new NeuralLayer(m_numOutputs, m_numInputs);
            mListLayers.add(outLayer2);
        }
    }

    // calculate the output from a given set of inputs
    public List<Double> feedForward(List<Double> pInputs) {
        List<Double> outputs = new ArrayList();

        int weightIndex;

        if (pInputs.size() != m_numInputs) {
            return outputs; // return empty
        }

        // at least once, plus the number of hidden layers
        for (int i = 0; i < m_numHiddenLayers + 1; i++) {
            if (i > 0) {
                pInputs = new ArrayList(outputs);
            }
            outputs.clear();

            weightIndex = 0;

            // for each neuron sum the (inputs * corresponding weights
            // throw the total at our sigmoid function to get the output
            // loop through all neurons in each layer
            for (int j = 0; j < mListLayers.get(i).mNumNeurons; j++) {
                double weightedSum = 0.0;

                // number of inputs for this layer
                int numInputs = mListLayers.get(i).mListNeurons.get(j).m_numInputs;

                // for each weight in the nueron of this layer
                for (int k = 0; k < numInputs - 1; k++) {
                    weightedSum += mListLayers.get(i).mListNeurons.get(j).m_weights.get(k)
                            * pInputs.get(weightIndex++);
                }

                // add in the bias separatley (from the above loop)
                weightedSum += mListLayers.get(i).mListNeurons.get(j).m_weights.get(numInputs - 1) * mBias;

                // add the activation response to the list of outputs of each node
                outputs.add(sigmoid(weightedSum, mResponse));

                weightIndex = 0;
            }
            mListLayers.get(i).mListActivations = new ArrayList(outputs);
        }
        return outputs;
    }

    public double sigmoid(double netinput, double response) {
        return (1.0 / (1.0 + Math.pow(Math.E, (-1.0 * netinput / response))));
    }

    void backPropagate(double classIndex) {
        double error;
        double target;
        double activation;
        double weightValue;
        double weightedSum = 0;
        List<Double> listError = new ArrayList();

        // 1. calculate output error for each neuron in this layer
        //    starting at the output layer
        // for each node in this layer
        for (int i = 0; i < mListLayers.get(m_numHiddenLayers).mNumNeurons; i++) {
            if (i == classIndex) {
                target = 1.0;
            } else {
                target = 0.0;
            }
            activation = mListLayers.get(m_numHiddenLayers).mListActivations.get(i);
            error = activation * (1 - activation) * (activation - target);
            listError.add(error);
        }
        mListLayers.get(m_numHiddenLayers).mListError = new ArrayList(listError);

        // 2. calculate hidden error
        if (m_numHiddenLayers > 0) {
            for (int i = m_numHiddenLayers; i > 0; i--) {
                listError.clear();
                // for each node in this layer
                for (int j = 0; j < mListLayers.get(i - 1).mNumNeurons; j++) {
                    activation = mListLayers.get(i - 1).mListActivations.get(j);
                    for (int k = 0; k < mListLayers.get(i).mNumNeurons; k++) {
                        weightedSum += mListLayers.get(i).mListError.get(k) * mListLayers.get(i).mListNeurons.get(k).m_weights.get(j);
                    }
                    error = activation * (1 - activation) * weightedSum;
                    listError.add(error);
                }
                mListLayers.get(i - 1).mListError = new ArrayList(listError);
            }
        }
        for (int i = 0; i < m_numHiddenLayers + 1; i++) {
            for (int j = 0; j < mListLayers.get(i).mNumNeurons; j++) {
                int numInputs = mListLayers.get(i).mListNeurons.get(j).m_numInputs;
                for (int k = 0; k < numInputs - 1; k++) {
                    weightValue = mListLayers.get(i).mListNeurons.get(j).m_weights.get(k);
                    weightValue -= mLearningRate * mListLayers.get(i).mListError.get(j)
                            * mListLayers.get(i).mListActivations.get(j);
                    mListLayers.get(i).mListNeurons.get(j).m_weights.set(k, weightValue);
                }
            }
        }
    }
}