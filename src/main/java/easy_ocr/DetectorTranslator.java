package easy_ocr;

import ai.djl.modality.cv.*;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.*;
import ai.djl.translate.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.ndarray.FloatNdArray;

public class DetectorTranslator implements NoBatchifyTranslator<Image, BBox[]>{
	private final int maxLength;
	private float scale;
	private int height;
	private int width;
	private static final int net_scale = 2;
	private static final double PIXEL_REGION_THRESHOLD = 0.4;
	private static final double PIXEL_AFFINITY_THRESHOLD = 0.4;
	private static final double PIXEL_SCORE_THRESHOLD = 0.4;
	private static final int MAXVAL_THRESHOLD_TYPE = 1;
	private static final int SIZE_THRESHOLD = 10;
	
	private static final float[] MEANS = {0.485f, 0.456f, 0.406f};
	private static final float[] VARIANCES = {0.229f, 0.224f, 0.225f};
	
	private static final double DETECTION_THRESHOLD = 0.7;
	private static final double BBOX_LEFT_VAL_THRESHOLD = 1e6;
	private static final double BBOX_TOP_VAL_THRESHOLD = 1e6;
	private static final double BBOX_RIGHT_VAL_THRESHOLD = 0;
	private static final double BBOX_BOTTOM_VAL_THRESHOLD = 0;

    /**
     * Creates the {@link DetectorTranslator} instance.
     *
     * @param arguments the arguments for the translator
     */
    public DetectorTranslator(Map<String, ?> arguments) {
        maxLength = ArgumentsUtil.intValue(arguments, "maxLength", 960);
    }

    /** {@inheritDoc} */
    @Override
    public BBox[] processOutput(TranslatorContext ctx, NDList list) {
        NDArray prediction = list.get(0);
        Mat regionMap = new Mat(height, width, CvType.CV_32FC1);
		Mat affinityMap = new Mat(height, width, CvType.CV_32FC1);
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				regionMap.put(i, j, prediction.getFloat(0, i, j, 0));
				affinityMap.put(i, j, prediction.getFloat(0, i, j, 1));
			}
		}
		
		Mat regionScore = new Mat(height, width, CvType.CV_32FC1);
		Mat affinityScore = new Mat(height, width, CvType.CV_32FC1);
		Mat score = new Mat(height, width, CvType.CV_32FC1);

		// Thresholding used for image segmentation, more precisely the pixels greater than
		// a given threshold will be replaced with a standard value. We are looking for pixels
		// where their "probability" of being in one of the letters is more then certain threshold
		Imgproc.threshold(regionMap, regionScore, PIXEL_REGION_THRESHOLD, MAXVAL_THRESHOLD_TYPE, Imgproc.THRESH_BINARY);
		Imgproc.threshold(affinityMap, affinityScore, PIXEL_AFFINITY_THRESHOLD, MAXVAL_THRESHOLD_TYPE, Imgproc.THRESH_BINARY);
		
		
		// There we are using threshold as a substitute for np.clip which should've kept values of score 
		// either 1 or 0
		Core.add(regionScore, affinityScore, score);
		Imgproc.threshold(score, score, PIXEL_SCORE_THRESHOLD, MAXVAL_THRESHOLD_TYPE, Imgproc.THRESH_BINARY);
		score.convertTo(score, CvType.CV_8UC1);

		Mat labels = new Mat(height, width, CvType.CV_32S), stats = new Mat(), centroids = new Mat();
        int nComponents = Imgproc.connectedComponentsWithStats(score, labels, stats, centroids);
        ArrayList<BBox> boxes = new ArrayList<>();
        
        for (int componentId = 1; componentId < nComponents; ++componentId) {
        	// If the detected component is really small we are just skipping it
            int size = (int)(stats.get(componentId, Imgproc.CC_STAT_AREA)[0]);
            
            if (size < SIZE_THRESHOLD) 
            	continue;
            
            // There we are trying to see if the maximum score in a component is big enough to
            // think that we are dealing with a text
            double mx = 0.0;
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                	if (labels.get(i, j)[0] == componentId) {
                		mx = Math.max(regionMap.get(i, j)[0], mx);
                	}
                }
            }

            if (mx < DETECTION_THRESHOLD)
				continue;
            
            // We are trying to create a boundary for the letter, so we are putting 255 only to the points
            // where one of the scores is 0 which still in the component
			Mat segmap = new Mat(height, width, CvType.CV_32FC1);
			for (int i = 0; i < height; ++i) {
				for (int j = 0; j < width; ++j) {
					if ((labels.get(i, j)[0] == componentId)
							&& (regionScore.get(i, j)[0] * affinityScore.get(i, j)[0] == 0)) {
						segmap.put(i, j, 255);
					} else {
						segmap.put(i, j, 0);
					}
				}
			}
            
			// Dilating a potential boundary so we will get better contour
			int w = (int) stats.get(componentId, Imgproc.CC_STAT_WIDTH)[0];
			int h = (int) stats.get(componentId, Imgproc.CC_STAT_HEIGHT)[0];
			int nIter = (int) (Math.sqrt(size * Math.min(w, h) / (w * h)) * 2);
			Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(nIter + 1, nIter + 1));
			Imgproc.dilate(segmap, segmap, kernel);

			Mat hierarchy = new Mat();
			List<MatOfPoint> contours = new ArrayList<>();
			
			// Finding the contours on a segmentation map using standard function
			// We are using only external contours so that it runs faster and simple approximation
			// because we need only the bounding box
			segmap.convertTo(segmap, CvType.CV_8UC1);
			Imgproc.findContours(segmap, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

			List<MatOfPoint2f> newContours = new ArrayList<>();
			for (MatOfPoint point : contours) {
				MatOfPoint2f newPoint = new MatOfPoint2f(point.toArray());
				newContours.add(newPoint);
			}

			// Using the best contour to create a bounding box
            MatOfPoint2f contour = newContours.get(0);
			Mat box = new Mat();
			// Creating the bounding box of minimal area from the contour points (just in case)
			Imgproc.boxPoints(Imgproc.minAreaRect(contour), box);
			// Here we are basically just normalising coordinates
			Mat box01 = new Mat();
			Mat box12 = new Mat();
			Core.subtract(box.row(0), box.row(1), box01);
			Core.subtract(box.row(1), box.row(2), box12);
			w = frobeniusNorm(box01);
			h = frobeniusNorm(box12);
			// Check to see if we have a diamond
			double box_ratio = Math.max(w, h) / (Math.min(w, h) + 1e-5);
			BBox bbox;

            if (Math.abs(1 - box_ratio) <= 0.1) {
				// Working with the diamond
				double lft = BBOX_LEFT_VAL_THRESHOLD;
				double rgt = BBOX_RIGHT_VAL_THRESHOLD;
				double tp = BBOX_TOP_VAL_THRESHOLD;
				double btm = BBOX_BOTTOM_VAL_THRESHOLD;

				for (int i = 0; i < contour.width(); ++i) {
					lft = Math.min(lft, contour.get(i, 0)[0]);
					rgt = Math.max(rgt, contour.get(i, 0)[0]);
					tp = Math.min(tp, contour.get(i, 0)[1]);
					btm = Math.max(btm, contour.get(i, 0)[1]);
				}
				bbox = new BBox(lft * net_scale / scale, rgt * net_scale / scale, tp * net_scale / scale, btm * net_scale / scale);
			} else {
				bbox = new BBox(box, net_scale / scale);
			}
			boxes.add(bbox);
        }
        BBox[] result = new BBox[boxes.size()];
        result = boxes.toArray(result);
	    return result;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDArray img = input.toNDArray(ctx.getNDManager());
        height = input.getHeight();
        width = input.getWidth();
        int[] hw = scale(height, width, maxLength);
		System.out.println("Current scaling: " + scale);

        height = hw[0] / net_scale;
        width = hw[1] / net_scale;
        img = NDImageUtils.resize(img, hw[1], hw[0]);
        img = NDImageUtils.toTensor(img);
        img = NDImageUtils.normalize(img, MEANS, VARIANCES);
        img = img.expandDims(0);
        return new NDList(img);
    }
    
    private int[] scale(int h, int w, int max) {
        int localMax = Math.max(h, w);
        scale = 1.0f;
        if (max < localMax) {
            scale = max * 1.0f / localMax;
        }
        // paddle model only take 32-based size
        return resize32(h * scale, w * scale);
    }

    private int[] resize32(double h, double w) {
        double min = Math.min(h, w);
        if (min < 32) {
            h = 32.0 / min * h;
            w = 32.0 / min * w;
        }
        int h32 = (int) h / 32;
        int w32 = (int) w / 32;
        return new int[] {h32 * 32, w32 * 32};
    }
    
    private static int frobeniusNorm(Mat mat) {
        int sumOfSquare = 0;
        for (int i = 0; i < mat.height(); i++){
            for (int j = 0; j < mat.width(); j++){
            	sumOfSquare += (int)Math.pow(mat.get(i, j)[0], 2);
            }
        }
        return (int)Math.sqrt(sumOfSquare);
    }
}