package experimentshell;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * @author mormon
 */
public class NeuralNetworkClassifier extends Classifier {
    private Instances iData;

    private NeuralNetwork nn;
    private List<Double> mOutputs;

    public NeuralNetworkClassifier() {
        this.mOutputs = new ArrayList();
    }

    @Override
    public void buildClassifier(Instances i) throws Exception {
        iData = i;
        makeNeuralNetwork();
    }

    
    public void makeNeuralNetwork() {
        nn = new NeuralNetwork(iData.numAttributes() - 1, iData.numClasses(), 1, 4);
        List<Double> attributeValues = new ArrayList();

        for (int i = 0; i < iData.numInstances(); i++) {
            for (int j = 0; j < iData.instance(i).numAttributes() - 1; j++) {
                attributeValues.add(iData.instance(i).value(j));
            }
            nn.feedForward(attributeValues);
            nn.backPropagate(iData.instance(i).value(iData.instance(i).classIndex()));
            attributeValues.clear();
        }
    }

    @Override
    public double classifyInstance(Instance instance) {
        List<Double> attributeValues = new ArrayList();
        for (int i = 0; i < instance.numAttributes() - 1; i++) {
            attributeValues.add(instance.value(i));
        }

        // call feedforward
        mOutputs = nn.feedForward(attributeValues);

        if (mOutputs.size() > 0) {
            return mOutputs.indexOf(Collections.max(mOutputs));
        } else {
            return 0;
        }
    }
}