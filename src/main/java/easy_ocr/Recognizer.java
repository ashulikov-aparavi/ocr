package easy_ocr;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.opencv.OpenCVImageFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

public class Recognizer {
	Predictor<RecognizerInput, RecognizerOutput> predictor;
	String currentModel;
	private static final float slope_ths = 0.1f;
	private static final float ycenter_ths = 0.5f;
	private static final float height_ths = 0.5f;
	private static final float width_ths = 0.5f;
	private static final float add_margin = 0.1f;
	
	private static final int MODEL_HEIGHT = 64;
	private static final int MIN_SIZE = 15;
	
	private static final HashMap<String, String> charSet = new HashMap<>();
	private static final String modelPath = "models/";

    static {
    	charSet.put("latin_gen2", " !\"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~ªÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿĀāĂăĄąĆćČčĎďĐđĒēĖėĘęĚěĞğĨĩĪīĮįİıĶķĹĺĻļĽľŁłŃńŅņŇňŒœŔŕŘřŚśŞşŠšŤťŨũŪūŮůŲųŸŹźŻżŽžƏƠơƯưȘșȚțə̇ḌḍḶḷṀṁṂṃṄṅṆṇṬṭẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ€");
    }
	 
    public Recognizer(String modelType) {
    	currentModel = modelType;
    	ConcurrentHashMap <String, String> translatorInput = new ConcurrentHashMap <String, String>();
    	translatorInput.put("charSet", supportedChars());
    	Criteria<RecognizerInput, RecognizerOutput> criteria = Criteria.builder()
				.setTypes(RecognizerInput.class, RecognizerOutput.class)
		        .optModelPath(Paths.get(modelPath + modelType))
		        .optTranslator(new RecognizerTranslator(translatorInput))
		        .optProgress(new ProgressBar()).build();

		try {
			ZooModel<RecognizerInput, RecognizerOutput> model = criteria.loadModel();
			this.predictor = model.newPredictor();
		} catch (ModelNotFoundException | MalformedModelException |IOException e) {
			e.printStackTrace();
		}
	}
    
	public Recognizer() {
		this("latin_gen2");
	}
	
	public Predictor<RecognizerInput, RecognizerOutput> getDetectorPredictor() {
		return this.predictor;
	}

	// For checking which char set we support during training the recognizer model
	public String supportedChars() {
		return charSet.get(currentModel);
	}

	private Image compute_ratio_and_resize(Mat img, int width,int height,int model_height) {
	    double ratio = width/1.0/height;
        Mat new_img = new Mat();
		Image res = null;
	    if (ratio<1.0) {
	        ratio = 1.0 / ratio;
	    }
        Imgproc.resize(img, new_img, new Size((int)(model_height*ratio), model_height), 0, 0, Imgproc.INTER_LANCZOS4);
        NDManager manager = NDManager.newBaseManager();
        NDArray full = manager.create(new Shape(3, model_height,(int)(model_height*ratio)));
		for (int i = 0; i < new_img.height(); i++) {
			for (int j = 0; j < new_img.width(); j++) {
				double[] pixel = new_img.get(i, j);
				full.setScalar(new NDIndex(0, i, j), (float) (pixel[0]));
				full.setScalar(new NDIndex(1, i, j), (float) (pixel[1]));
				full.setScalar(new NDIndex(2, i, j), (float) (pixel[2]));
			}
		}
		res = ImageFactory.getInstance().fromNDArray(full);
	    
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

	/***
	 * @param image
	 * @param bboxes
	 * @return list of predicted output words along with the bbox coordinates
	 * 
	 * This API is responsible for processing the image stream with the provided bounding boxes that
	 * generated from Detector model, crop and scale those boxes to a convenient shape and invoke 
	 * the recognizer model on each reformed bbox to get the set of predicted character set   
	 */

	public Prediction[] recognizeFromBBoxes(Mat image, BBox[] bboxes) {
		ArrayList<RecognizerInput> image_list = get_image_part(image, bboxes);
		Prediction[] recognitions = new Prediction[image_list.size()];
		for (int i = 0; i < image_list.size(); ++i) {
			try {
				RecognizerInput inp = image_list.get(i);
				recognitions[i] = new Prediction(inp.bbox);
				RecognizerOutput output = predictor.predict(inp);
				//System.out.println("Current score: " + output.score);
				if (output.score > 0.3) {
					recognitions[i].word = output.ans;
				}
				else {
					inp.contrast = true;
					RecognizerOutput new_output = predictor.predict(inp);
					//System.out.println("Score after contrast: " + output.score);
					recognitions[i].word = new_output.ans;
				}
			} catch (TranslateException e) {
				e.printStackTrace();
			}
		}
		return recognitions;
	}
}
