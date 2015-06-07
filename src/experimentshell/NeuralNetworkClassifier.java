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
   public void buildClassifier(Instances data) throws Exception {       
       iData = data;
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
   public double classifyInstance(Instance inst) {
       List<Double> attrValues = new ArrayList();
      
       for (int i = 0; i < inst.numAttributes() - 1; i++) {
           attrValues.add(inst.value(i));
       }
       
       // call feedForward
       mOutputs = nn.feedForward(attrValues);
      
       if (mOutputs.size() > 0)
           return mOutputs.indexOf(Collections.max(mOutputs));
       else 
           return 0;
   }
}