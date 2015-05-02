/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experimentshell;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 *
 * @author mormon
 */
public class ExperimentShell {

    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
        String file = "C:\\Users\\mormon\\Documents\\NetBeansProjects\\experimentShell\\src\\Data.csv";
        
        // read in the CSV file and parse it
        DataSource source = new DataSource(file); 
        Instances data = source.getDataSet();
        
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        
        // randomize the data 
        data.randomize(new Random(1));
        
        /****************************************************
         * Split the data up into its proper sets. #3
         ***************************************************/
        // set a filter to pull out the 70%
        RemovePercentage remove = new RemovePercentage();
        remove.setPercentage(70);
        
        // split the data for training set
        remove.setInputFormat(data);
        Instances training =  Filter.useFilter(data, remove);

        // split the data for test set
        remove.setInvertSelection(true);
        remove.setInputFormat(data);
        Instances test = Filter.useFilter(data, remove);
        
        // Standardize the data
        Standardize filter = new Standardize();
        filter.setInputFormat(training);
        
        Instances newTest = Filter.useFilter(test, filter);
        Instances newTraining = Filter.useFilter(training, filter);
        
        /*
         * The Nearest Neighbor implementation 
         */
        Classifier knn = new knn();
         knn.buildClassifier(newTraining);
         Evaluation eval = new Evaluation(newTraining);
         eval.evaluateModel(knn, newTest);
         
         System.out.println(eval.toSummaryString("***** Overall results: *****", false));
    }
}