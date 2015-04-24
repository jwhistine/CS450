/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experimentshell;
import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
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
        String file;
        
        file = "C:\\Users\\mormon\\Documents\\NetBeansProjects\\experimentShell\\src\\Data.csv";
        
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
        RemovePercentage filter = new RemovePercentage();
        filter.setPercentage(70);
        
        // split the data for training set
        filter.setInputFormat(data);
        Instances training =  Filter.useFilter(data, filter);

        // split the data for test set
        filter.setInvertSelection(true);
        filter.setInputFormat(data);
        Instances test = Filter.useFilter(data, filter);
        
        System.out.println("Training Set: ");
        // print out the training set
        for (Instance training1 : training) {
            System.out.println(training1);
        }
        
        System.out.println("Test set: ");
        // print the test set
        for (Instance test1 : test) {
            System.out.println(test1);
        }
    }
}