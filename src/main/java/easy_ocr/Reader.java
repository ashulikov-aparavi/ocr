package easy_ocr;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;

import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;

import org.opencv.core.Core;
// import org.opencv.core.CvType;

// Java code guideline: https://www.oracle.com/java/technologies/javase/codeconventions-namingconventions.html

public class Reader {
	// private static Pipeline pipelineInstance = null; // Singleton instance
	private Detector boundingBoxDetector;
	private Recognizer recognizerInstance;
	private double scale;
	private int maxSize;

	private static final int IMAGE_OPS_SCALING = 0; // Macro for image scaling
	private static final int MAXIMUM_NUMBER_OF_LINES = 75; // MS answers says that standard number of lines
	                                                       // per page is 46. We are using more generous 75.

	public Reader(String[] lang_list, Boolean gpu, String detect_network, Boolean quantize, Boolean cudnn_benchmark, float sc, int ms) {
		this.initDetector(detect_network);
		this.initRecognizer();
		this.setScaling(sc);
		this.setMaxImgDimension(ms);
	}

	public Reader() {
        System.out.println("Invoking default TFPipeline Ctor ........... ");
		this.initDetector("craft");
		this.initRecognizer();
		this.setScaling(1.0); // default scaling
		this.setMaxImgDimension(2560); // default image dimension
	}

	public void initDetector(String detectorType) {
		this.boundingBoxDetector = new Detector();
	}

	public void initRecognizer(String recognizerType) {
		this.recognizerInstance = new Recognizer(recognizerType);
	}
	
	public void initRecognizer() {
		this.recognizerInstance = new Recognizer();
	}

	public void setScaling(double scale) {
		this.scale = scale;
	}

	public void setMaxImgDimension(int maxSize) {
		this.maxSize = maxSize;
	}

	// Mainly added the following four getter methods for unit testing
	public Detector getDetectorInstance() {
		return this.boundingBoxDetector;
	}

	public Recognizer getRecognizerInstance() {
		return this.recognizerInstance;
	}

	public double getScaling() {
		return this.scale;
	}
	
	public int getMaxImgDimension() {
		return this.maxSize;
	}
	
    public static void sort(ArrayList<Prediction> list, int num) {
        list.sort((o1, o2)-> Double.compare(o1.getCustomProperty(num), o2.getCustomProperty(num)));
    }
    
    public Prediction[] connectWords(Prediction[] ans, double part_h) {
    	ArrayList<Prediction> ans_v = new ArrayList<Prediction>();
    	for (int i = 0; i < ans.length; i++) {
	    	if ((ans[i].bbox != null)&&(ans[i].bbox.points.size() > 0)) {
	    		ans_v.add(ans[i]);
	    	}
	    }
    	if (ans_v.size() == 0) {
    		return new Prediction[0];
    	}
	    sort(ans_v, 0);
	    double low_y = ans_v.get(0).bbox.points.get(0).y;
	    int cnt = 0;
	    ArrayList<Prediction> ans_tmp = new ArrayList<Prediction>();
	    ans_tmp.add(ans_v.get(0));
	    for (int i = 1; i < ans_v.size(); i++) {
	    	if (ans_v.get(i).bbox.points.get(0).y - low_y > part_h) {
	    		cnt += 1;
	    		low_y = ans_v.get(i).bbox.points.get(0).y;
	    	}
	    }
	    low_y = ans_v.get(0).bbox.points.get(0).y;
	    Prediction[] result = new Prediction[cnt+1];
	    cnt = 0;
	    for (int i = 1; i < ans_v.size(); i++) {
	    	if (ans_v.get(i).bbox.points.get(0).y - low_y > part_h) {
	    		sort(ans_tmp, 1);
	    		double left=ans_tmp.get(0).bbox.points.get(0).x;
	    		double top=ans_tmp.get(0).bbox.points.get(0).y;
	    		double right=ans_tmp.get(0).bbox.points.get(2).x;
	    		double bottom=ans_tmp.get(0).bbox.points.get(2).y;
	    		String word = ans_tmp.get(0).word;
	    		for (int j = 1; j < ans_tmp.size(); j++) {
	    			left = Math.min(left, ans_tmp.get(j).bbox.points.get(0).x);
	    			top = Math.min(top, ans_tmp.get(j).bbox.points.get(0).y);
	    			right = Math.max(right, ans_tmp.get(j).bbox.points.get(2).x);
	    			bottom = Math.max(bottom, ans_tmp.get(j).bbox.points.get(2).y);
	    			word += " " + ans_tmp.get(j).word;
	    		}
	    		result[cnt] = new Prediction(new BBox(left, right, top, bottom));
	    		result[cnt].word = word;
	    		low_y = ans_v.get(i).bbox.points.get(0).y;
	    		cnt += 1;
	    		ans_tmp = new ArrayList<Prediction>();
	    	}
	    	ans_tmp.add(ans_v.get(i));
	    }
	    sort(ans_tmp, 1);
		double left=ans_tmp.get(0).bbox.points.get(0).x;
		double top=ans_tmp.get(0).bbox.points.get(0).y;
		double right=ans_tmp.get(0).bbox.points.get(2).x;
		double bottom=ans_tmp.get(0).bbox.points.get(2).y;
		String word = ans_tmp.get(0).word;
		for (int j = 1; j < ans_tmp.size(); j++) {
			left = Math.min(left, ans_tmp.get(j).bbox.points.get(0).x);
			top = Math.min(top, ans_tmp.get(j).bbox.points.get(0).y);
			right = Math.max(right, ans_tmp.get(j).bbox.points.get(2).x);
			bottom = Math.max(bottom, ans_tmp.get(j).bbox.points.get(2).y);
			word += " " + ans_tmp.get(j).word;
		}
		result[cnt] = new Prediction(new BBox(left, right, top, bottom));
		result[cnt].word = word;
		cnt += 1;
	    return result;
    }

	// This static method creates the Piepline instance only once
	// used by multiple threads
	// public static Pipeline getInstance() {
	// 	if (pipelineInstance == null) {
	// 		System.out.println("Tensorflow OCR Pipeline is loaded for the first time");
	// 		pipelineInstance = new Pipeline();
	// 	} else {
	// 		System.out.println("Already Pipeline singleton has been created");
	// 	}
	// 	return pipelineInstance;
	// }
	
	// private Mat imgResizeOnScaling(Mat image) {
	// 	Size sz = new Size((int) (image.size().width * this.scale), (int) (image.size().height * this.scale));
	// 	Mat resizedImage = new Mat();
	// 	Imgproc.resize(image, resizedImage, sz);
	// 	return resizedImage;
	// }

	// private Mat imgRefinedOnRotation(Mat image) {
	// 	Size rotSize = new Size(this.rotationSize, this.rotationSize);
	// 	Mat rotatedImage = new Mat();
	// 	Imgproc.resize(image, rotatedImage, rotSize);
	// 	int rotation = this.rotatorInstance.apply(rotatedImage);
	// 	System.out.println(rotation);
	// 	for (int i = 0; i < rotation; i++) {
	// 		Core.rotate(image, image, Core.ROTATE_90_CLOCKWISE);
	// 	}

	// 	// HighGui.imshow("Image", image); // For debugging
	// 	// HighGui.waitKey();
	// 	return image;
	// }

	public Prediction[] recognize(Mat inputImage, String image_path) {
		System.out.println("Current image path: " + image_path);
		//Imgproc.cvtColor(inputImage, inputImage, Imgproc.COLOR_RGB2GRAY); 
	    BBox[] bboxes = boundingBoxDetector.detectBBoxes(image_path);
		Prediction[] ans = recognizerInstance.recognizeFromBBoxes(inputImage, bboxes);
		double part_h = inputImage.height() / MAXIMUM_NUMBER_OF_LINES; // This constant is the height each line should have
	    
	    return connectWords(ans, part_h);
	}
}

