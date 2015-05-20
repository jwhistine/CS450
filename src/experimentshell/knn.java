package experimentshell;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import static java.lang.Math.abs;
import static java.lang.Math.pow;
import java.util.Map;
import java.util.TreeMap;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author mormon
 */
public class knn extends Classifier {    
    
    Integer k = 8;
    Instances data;
    
    @Override
   public void buildClassifier(Instances i) throws Exception {       
       data = i;
   }
   
   @Override
   public double classifyInstance(Instance instance) throws Exception {
       // sorting happens automatically with a TreeMap
       Map<Integer, Double> distances = new TreeMap<>();
       int distance = 0;
       
       // iterate through the data and determine the distance
       for (int i = 0; i < data.numInstances(); i++) {
           for (int j = 0; j < data.numAttributes(); j++) {
               // this is the Manhattan distance
               distance += abs((int)(data.instance(i).value(j) - instance.value(j)));
               // the Euclidean distance
               // distance += pow((int)(data.instance(i).value(j) - instance.value(j)), 2);
           }   
           distances.put(distance, data.instance(i).classValue());
           distance = 0;
       }
       
       // find the majority class of the nearest neighbor
       int count = 0;
       double tempClass;
       int tally[] = new int[data.numClasses()];
       // use k instances to find the majority class and assign that instance
       for (Map.Entry<Integer, Double> entry : distances.entrySet()) {
           if (count >= k) 
               break;
           tempClass = entry.getValue();
           tally[(int)tempClass]++;
           count++;
       }
       
       int maxIndex = 0;
       int majority;
       for (int i = 0; i < data.numClasses(); i++) {
           majority = tally[i];
           if (majority > tally[maxIndex]) {
               maxIndex = i;
           }
       }
       
       return maxIndex;
   }
}