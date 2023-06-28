package easy_ocr;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.opencv.OpenCVImageFactory;

import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
// import org.opencv.core.CvType;
import org.opencv.core.CvType;

// Java code guideline: https://www.oracle.com/java/technologies/javase/codeconventions-namingconventions.html

public class Reader {
	// private static Pipeline pipelineInstance = null; // Singleton instance
	private Detector boundingBoxDetector;
	private ScriptIdentificator scriptIdentificationHead;
	private Recognizer[] recognizerHeads;
	private double scale;
	private int maxSize;

	private static final float slope_ths = 0.1f;
	private static final float ycenter_ths = 0.5f;
	private static final float height_ths = 0.5f;
	private static final float width_ths = 0.5f;
	private static final float add_margin = 0.1f;
	
	private static final int MODEL_HEIGHT = 64;
	private static final int MIN_SIZE = 10;
	private static final int MAXIMUM_NUMBER_OF_LINES = 75; // MS answers says that standard number of lines
	                                                       // per page is 46. We are using more generous 75.

	private static final String[] LANGS = {"latin_gen2", "cyrillic_gen2"};

	public Reader(String[] lang_list, Boolean gpu, String detect_network, Boolean quantize, Boolean cudnn_benchmark, float sc, int ms) {
		this.initDetector(detect_network);
		this.initScriptIdentificator();
		this.initRecognizers();
		this.setScaling(sc);
		this.setMaxImgDimension(ms);
	}

	public Reader() {
        System.out.println("Invoking default TFPipeline Ctor ........... ");
		this.initDetector("craft");
		this.initScriptIdentificator();
		this.initRecognizers();
		this.setScaling(1.0); // default scaling
		this.setMaxImgDimension(2560); // default image dimension
	}

	public void initDetector(String detectorType) {
		this.boundingBoxDetector = new Detector();
	}
	
	public void initScriptIdentificator() {
		this.scriptIdentificationHead = new ScriptIdentificator();
	}

	public void initRecognizers(String recognizerType) {
		this.recognizerHeads = new Recognizer[1];
		this.recognizerHeads[0] = new Recognizer(recognizerType);
	}
	
	public void initRecognizers() {
		this.recognizerHeads = new Recognizer[LANGS.length];
		for (int i = 0; i < LANGS.length; ++i) {
			this.recognizerHeads[i] = new Recognizer(LANGS[i]);
		}
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

	public Recognizer getRecognizerInstance(int n) {
		return this.recognizerHeads[n];
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
    
    private Image compute_ratio_and_resize(Mat img, int width,int height,int model_height) {
	    double ratio = width/1.0/height;
        Mat new_img = new Mat();
	    if (ratio<1.0) {
	        ratio = 1.0 / ratio;
	    }
        Imgproc.resize(img, new_img, new Size((int)(model_height*ratio), model_height), 0, 0, Imgproc.INTER_LANCZOS4);
        Image res = OpenCVImageFactory.getInstance().fromImage(new_img);
	    return res;
	}
	
	private Mat four_point_transform(Mat image, List<Point> box) {
	    List<Point> rect = box;

	    double widthA = Math.sqrt(((rect.get(2).x- rect.get(3).x)*(rect.get(2).x- rect.get(3).x)) + 
	    		                  ((rect.get(2).y - rect.get(3).y)*(rect.get(2).y - rect.get(3).y)));
	    double widthB = Math.sqrt(((rect.get(0).x- rect.get(1).x)*(rect.get(0).x- rect.get(1).x)) + 
                                  ((rect.get(0).y - rect.get(1).y)*(rect.get(0).y - rect.get(1).y)));
	    int maxWidth = Math.max((int)widthA, (int)widthB);

	    // compute the height of the new image, which will be the
	    // maximum distance between the top-right and bottom-right
	    // y-coordinates or the top-left and bottom-left y-coordinates
	    double heightA = Math.sqrt(((rect.get(2).x- rect.get(1).x)*(rect.get(2).x- rect.get(1).x)) + 
                				   ((rect.get(2).y - rect.get(1).y)*(rect.get(2).y - rect.get(1).y)));
	    double heightB = Math.sqrt(((rect.get(0).x- rect.get(3).x)*(rect.get(0).x- rect.get(3).x)) + 
                				   ((rect.get(0).y - rect.get(3).y)*(rect.get(0).y - rect.get(3).y)));
	    int maxHeight = Math.max((int)heightA, (int)heightB);
	    List<Point> points = new ArrayList<>();
		points.add(new Point(0, 0));
		points.add(new Point(maxWidth - 1, 0));
		points.add(new Point(maxWidth - 1, maxHeight - 1));
		points.add(new Point(0, maxHeight - 1));
		MatOfPoint mPoints = new MatOfPoint();
		mPoints.fromList(points);
		Mat M = new Mat();
		Mat mPointsF = (Mat) mPoints;
		mPointsF.convertTo(mPointsF, CvType.CV_32F);
		MatOfPoint mRect = new MatOfPoint();
		mRect.fromList(rect);
		Mat mRectF = (Mat) mRect;
		mRectF.convertTo(mRectF, CvType.CV_32F);
		M = Imgproc.getPerspectiveTransform(mRectF, mPointsF);
		Mat crop = new Mat();
		Imgproc.warpPerspective(image, crop, M, new Size(maxWidth, maxHeight));

	    return crop;
	}
	
	private ArrayList<RecognizerInput> get_image_part(Mat image, BBox[] bbox) {
		ArrayList<double[]> horizontal_list = new ArrayList<double[]>();
		ArrayList<List<Point>> free_list = new ArrayList<List<Point>>();
		ArrayList<ArrayList<double[]>> combined_list = new ArrayList<ArrayList<double[]>>();
		for (int i = 0; i < bbox.length; ++i) {
			List<Point> poly = bbox[i].points;
	        double slope_up = (poly.get(1).y-poly.get(0).y)/Math.max(10, (poly.get(1).x-poly.get(0).x));
	        double slope_down = (poly.get(2).y-poly.get(3).y)/Math.max(10, (poly.get(2).x-poly.get(3).x));
	        if (Math.max(Math.abs(slope_up), Math.abs(slope_down)) < slope_ths) {
	        	double[] tmp_point = {bbox[i].minX, bbox[i].maxX, bbox[i].minY, bbox[i].maxY, 0.5*(bbox[i].minY+bbox[i].maxY), bbox[i].maxY-bbox[i].minY};
	            horizontal_list.add(tmp_point);
	        }
	        else {
	            int margin = (int)(1.44*add_margin*Math.min(bbox[i].width, bbox[i].height));
	            double theta13 = Math.abs(Math.atan((poly.get(0).y-poly.get(2).y)/Math.max(10, (poly.get(0).x-poly.get(2).x))));
	            double theta24 = Math.abs(Math.atan((poly.get(1).y-poly.get(3).y)/Math.max(10, (poly.get(1).x-poly.get(3).x))));
	            List<Point> points = new ArrayList<>();
	    		points.add(new Point(poly.get(0).x - Math.cos(theta13)*margin, poly.get(0).y - Math.sin(theta13)*margin));
	    		points.add(new Point(poly.get(1).x + Math.cos(theta24)*margin, poly.get(1).y - Math.sin(theta24)*margin));
	    		points.add(new Point(poly.get(2).x + Math.cos(theta13)*margin, poly.get(2).y + Math.sin(theta13)*margin));
	    		points.add(new Point(poly.get(3).x - Math.cos(theta24)*margin, poly.get(3).y + Math.sin(theta24)*margin));
	            free_list.add(points);
	        }
		}
        horizontal_list.sort((o1, o2) -> Double.compare(o1[4], o2[4]));
	    // combine box
        if (horizontal_list.size() > 0) {
        	ArrayList<double[]>new_box = new ArrayList<double[]>(Arrays.asList(horizontal_list.get(0)));
        	ArrayList<Double> b_height = new ArrayList<Double>(Arrays.asList(horizontal_list.get(0)[5]));
            ArrayList<Double> b_ycenter = new ArrayList<Double>(Arrays.asList(horizontal_list.get(0)[4]));
    	    for (int i = 1; i < horizontal_list.size(); ++i) {
	            // comparable height and comparable y_center level up to ths*height
	            if (Math.abs(b_ycenter.stream().mapToDouble(d -> d).average().orElse(0.0) - horizontal_list.get(i)[4]) 
	            		< ycenter_ths*b_height.stream().mapToDouble(d -> d).average().orElse(0.0)){
	            	b_height.add(horizontal_list.get(i)[5]);
		            b_ycenter.add(horizontal_list.get(i)[4]);
		            new_box.add(horizontal_list.get(i));
	            }
	            else {
	                b_height = new ArrayList<Double>(Arrays.asList(horizontal_list.get(i)[5]));
	                b_ycenter = new ArrayList<Double>(Arrays.asList(horizontal_list.get(i)[4]));
	                combined_list.add(new_box);
	                new_box = new ArrayList<double[]>(Arrays.asList(horizontal_list.get(i)));
	            }
    	    }
    	    combined_list.add(new_box);
        }
        
	    // merge list use sort again
        for (int i = 0; i < combined_list.size(); ++i) {
	        if (combined_list.get(i).size() == 1) { // one box per line
	        	double[] box = combined_list.get(i).get(0);
	            int margin = (int)(add_margin*Math.min(box[1]-box[0],box[5]));
	            if (Math.max(box[1]+margin-box[0]+margin, box[3]+margin-box[2]+margin) > MIN_SIZE) {
	            	free_list.add(new BBox(box[0]-margin, box[1]+margin, box[2]-margin, box[3]+margin).points);
	            }
	        }
	        else { // multiple boxes per line
	        	ArrayList<double[]> boxes = combined_list.get(i);
	        	boxes.sort((o1, o2) -> Double.compare(o1[0], o2[0]));
	        	ArrayList<ArrayList<double[]>> merged_box = new ArrayList<ArrayList<double[]>>();
	        	
	        	ArrayList<Double> b_height = new ArrayList<Double>(Arrays.asList(boxes.get(0)[5]));
	        	double x_max = boxes.get(0)[1];
	        	ArrayList<double[]> new_box = new ArrayList<double[]>(Arrays.asList(boxes.get(0)));
	            for (int j = 0; j < boxes.size(); ++j) {
	            	double mean = b_height.stream().mapToDouble(d -> d).average().orElse(0.0);
                    if ((mean - boxes.get(j)[5] < height_ths*mean) && (boxes.get(j)[0]-x_max < width_ths *(boxes.get(j)[3]-boxes.get(j)[2]))) { // merge boxes
                        b_height.add(boxes.get(j)[5]);
                        new_box.add(boxes.get(j));
                    }
                    else {
                    	b_height = new ArrayList<Double>(Arrays.asList(boxes.get(j)[5]));
                        merged_box.add(new_box);
                        new_box = new ArrayList<double[]>(Arrays.asList(boxes.get(j)));
                    }
                    x_max = boxes.get(j)[1];
	            }
	            if (new_box.size() > 0) {
	            	merged_box.add(new_box);
	            }

	            for (int j = 0; j < merged_box.size(); ++j) {
	                if (merged_box.get(j).size() > 1) { // There's an adjacent box in same line
	                	double x_min = 1e6, y_min = 1e6, y_max = 0;
	                	x_max = 0;
	                	for (int k = 0; k < merged_box.get(j).size(); ++k) {
	                		x_min = Math.min(x_min, merged_box.get(j).get(k)[0]);
	                		x_max = Math.max(x_max, merged_box.get(j).get(k)[1]);
	                		y_min = Math.min(y_min, merged_box.get(j).get(k)[2]);
	                		y_max = Math.max(y_max, merged_box.get(j).get(k)[3]);
	                	}

	                    double box_width = x_max - x_min;
	                    double box_height = y_max - y_min;
	                    double margin = (int)(add_margin * (Math.min(box_width, box_height)));
	                    if (Math.max(box_width + 2*margin, box_height + 2*margin) > MIN_SIZE) {
	                    	free_list.add(new BBox(x_min-margin, x_max+margin, y_min-margin, y_max+margin).points);
	                    }
	                }
	                else{ // non adjacent box in same line
	                    double box_width = merged_box.get(j).get(0)[1] - merged_box.get(j).get(0)[0];
	                    double box_height = merged_box.get(j).get(0)[3] - merged_box.get(j).get(0)[2];
	                    double margin = (int)(add_margin * (Math.min(box_width, box_height)));
	                    
	                    if (Math.max(box_width + 2*margin, box_height + 2*margin) > MIN_SIZE) {
		                    free_list.add(new BBox(merged_box.get(j).get(0)[0]-margin,
		                    		               merged_box.get(j).get(0)[1]+margin,
		                    		               merged_box.get(j).get(0)[2]-margin,
		                    		               merged_box.get(j).get(0)[3]+margin).points);
	                    }
	                }
	            }
	        }
        }
		ArrayList <RecognizerInput> image_list = new ArrayList<RecognizerInput>();
	    for(int i = 0; i < free_list.size(); ++i) {
	    	List<Point> box = free_list.get(i);
	        Mat transformed_img = four_point_transform(image, box);
	        //HighGui.imshow("image", transformed_img);
	        //HighGui.waitKey();
	        double ratio;
	        if (transformed_img.width()<transformed_img.height()) {
	        	ratio = transformed_img.height()/1.0/transformed_img.width();
	        }
	        else {
	        	ratio = transformed_img.width()/1.0/transformed_img.height();
	        }
	        int new_width = (int)(MODEL_HEIGHT*ratio);
	        if (new_width != 0) {
	            Image crop_img = compute_ratio_and_resize(transformed_img,transformed_img.width(),transformed_img.height(),MODEL_HEIGHT);
	            image_list.add(new RecognizerInput(new BBox(box), crop_img)); // box = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
	        }
	    }
	    return image_list;
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
//		long start = System.nanoTime();
		//Imgproc.cvtColor(inputImage, inputImage, Imgproc.COLOR_RGB2GRAY); 
	    BBox[] bboxes = boundingBoxDetector.detectBBoxes(image_path);
//		long detectorEnd = System.nanoTime();
		ArrayList<RecognizerInput> image_list = get_image_part(inputImage, bboxes);
		int[] scriptNumber = scriptIdentificationHead.getScript(image_list);
		Prediction[] ans = new Prediction[scriptNumber.length];
		for (int i = 0; i < scriptNumber.length; ++i) {
			Recognizer recognizerInstance = getRecognizerInstance(scriptNumber[i]);
			System.out.printf("Predicted script", LANGS[i]);
			ans[i] = recognizerInstance.recognizeFromBbox(image_list.get(i));
		}
//		long recognizerEnd = System.nanoTime();
//		long timeElapsedDetector = detectorEnd - start;
//		long minutes = (timeElapsedDetector / 1000000000) / 60;
//		long seconds = (timeElapsedDetector / 1000000000) % 60;
//		System.out.printf("Total Time taken for Detector: %d minutes, %d seconds%n", minutes, seconds);
//		long timeElapsedRecognizer = recognizerEnd - detectorEnd;
//		minutes = (timeElapsedRecognizer / 1000000000) / 60;
//		seconds = (timeElapsedRecognizer / 1000000000) % 60;
//		System.out.printf("Total Time taken for Recognizer: %d minutes, %d seconds%n", minutes, seconds);
		double part_h = inputImage.height() / MAXIMUM_NUMBER_OF_LINES; // This constant is the height each line should have
	    
	    return connectWords(ans, part_h);
	}
}

