package easy_ocr;
import java.io.File;
import java.io.FileWriter;   // Import the FileWriter class
import java.io.IOException;  // Import the IOException class to handle errors

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class App 
{
    public static void main( String[] args )
    {
    	File dir = new File("data/");
    	File[] directoryListing = dir.listFiles();
    	if (directoryListing != null) {
    		for (File child : directoryListing) {
    			nu.pattern.OpenCV.loadShared();
    	        Reader pipeline = new Reader();
    			FileWriter myWriter;
    	        Mat inputImage = Imgcodecs.imread(child.getAbsolutePath());
    			Prediction[] prediction_groups = pipeline.recognize(inputImage, child.getAbsolutePath());
    			try {
    				String name = child.getName();
    				myWriter = new FileWriter("ans/"+ name.substring(0, name.length() - 4) +".txt");
    				for (int i = 0; i < prediction_groups.length; i++) {
    					if (prediction_groups[i] != null) myWriter.write(prediction_groups[i].word + "\n");
    					else System.out.println(name.substring(0, name.length() - 4) + " " + Integer.toString(i));
    		        	//if (prediction_groups[i].bbox != null) System.out.println(prediction_groups[i].bbox.getMat().dump());
    		        }
    				myWriter.close();
    			} catch (IOException e) {
    				e.printStackTrace();
    			}
    		}
    	} 
    }
}
