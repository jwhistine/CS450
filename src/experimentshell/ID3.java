package experimentshell;

import java.util.Enumeration;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import java.util.Arrays;

/**
 *
 * @author mormon
 */
public class ID3 extends Classifier {

    Instances data;
    
   @Override
   public void buildClassifier(Instances i) throws Exception {       
       data = i;
       makeTree();
   }
   
   /*
    * CLASSIFY INSTANCE
    */
   @Override
   public double classifyInstance(Instance instance) throws Exception {
       return 0;
   }
   
   /*
    * MAKE TREE
    * This is not done.
    */
   private void makeTree() throws Exception { 
       Enumeration classes = data.classAttribute().enumerateValues();
       Enumeration featureNames = data.enumerateAttributes();
       if (data.numInstances() == 0)
       {
            return;
       }
       else if (data.numInstances() == data.classAttribute().numValues()) {
            return;
       }    
       double[] infoGain = new double[data.numAttributes()];

        // calculate the information gain for each attribute
        Enumeration attributesEnum = data.enumerateAttributes();
        while (attributesEnum.hasMoreElements()) {
            Attribute attribute = (Attribute) attributesEnum.nextElement();
            infoGain[attribute.index()] = calcInfoGain(attribute.index());
        }
        Attribute mostInformativeAttribute
                = data.attribute(Utils.maxIndex(infoGain));
   }
   
   /*
    * CALC TOTAL ENTROPY
    */
   private double calcTotalEntropy() {
        int classIndex;
        double[] classCounts = new double[data.numClasses()];
        for (int i = 0; i < data.numInstances(); i++) {
            classIndex = (int) data.instance(i).classValue();
            classCounts[classIndex]++;
        }

        // calculate total entropy for class set
        double totalEntropy = 0;
        double classProbability;
        for (int i = 0; i < data.numClasses(); i++) {
            classProbability = classCounts[i] / data.numInstances();
            totalEntropy += calcEntropy(classProbability);
        }
        return totalEntropy;
    }
   
   /*
    * CALC INFO GAIN
    */
   private double calcInfoGain(int i) throws Exception {
        // find the total number of each class
        double totalEntropy = calcTotalEntropy();

        // calculate entropy for each attribute        
        double[] classValueCounts = new double[data.numClasses()];
        double attributeValueProbability;
        double attributeValueEntropy = 0;
        double classValueProbability;
        double totalAttributeValuesEntropy = 0;
        double gain;
        int attributeValueCount = 0;
        int attributeValueIndex;
        int classValueIndex;

        data.sort(i);

        for (int j = 0; j < data.numInstances(); j++) {
            if (j + 1 < data.numInstances() && data.instance(j).value(i) == data.instance(j + 1).value(i)) {
                classValueIndex = (int) data.instance(j).classValue();
                classValueCounts[classValueIndex]++;
                attributeValueCount++;
            } else {
                attributeValueProbability = attributeValueCount / (double) data.numInstances();
                for (int k = 0; k < data.numClasses(); k++) {
                    classValueProbability = classValueCounts[k] / attributeValueCount;
                    attributeValueEntropy += calcEntropy(classValueProbability);
                }
                totalAttributeValuesEntropy += (attributeValueEntropy * attributeValueProbability);
                attributeValueCount = 0;
                attributeValueEntropy = 0;
                Arrays.fill(classValueCounts, 0);
            }
        }

        gain = totalEntropy - totalAttributeValuesEntropy;
        return gain;
    }
   
   /*
    * CALC ENTROPY
    */
   private double calcEntropy(double p) {
       double result;
       if (p != 0)
           result = -p * Utils.log2(p);
       else
           result = 0; 
       
       return result;
   }
}