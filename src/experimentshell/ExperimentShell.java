/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experimentshell;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

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
        DataSource source = new DataSource(""); 
        Instances data = source.getDataSet();
        
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
       
        // print out the list to see what it looks like
        for (Instance data1 : data) {
            System.out.println(data1);
        }
    }
}
