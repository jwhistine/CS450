package experimentshell;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author mormon
 */
public class HardCodeClassifier extends Classifier {
   
   @Override
   public void buildClassifier(Instances data) throws Exception {       
   }
   
   @Override
   public double classifyInstance(Instance i) {
       return 0;
   }
}