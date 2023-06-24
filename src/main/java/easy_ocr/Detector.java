package easy_ocr;

import java.util.concurrent.ConcurrentHashMap;
import java.nio.file.*;
import java.io.IOException;

import ai.djl.*;
import ai.djl.inference.*;
import ai.djl.modality.cv.*;
import ai.djl.repository.zoo.*;
import ai.djl.translate.*;
import ai.djl.training.util.*;

public class Detector {
	Predictor<Image, BBox[]> predictor;
	// private static Logger logger = Logger.getLogger(Detector.class);
	
	private static final String modelPath = "models/";
	
	public Detector(String detectorType) {
		ConcurrentHashMap <String, String> translatorInput = new ConcurrentHashMap <String, String>();
		Criteria<Image, BBox[]> criteria = Criteria.builder()
				.setTypes(Image.class, BBox[].class)
		        .optModelPath(Paths.get(modelPath + detectorType))
		        .optTranslator(new DetectorTranslator(translatorInput))
		        .optProgress(new ProgressBar()).build();

		try {
			ZooModel<Image, BBox[]> model = criteria.loadModel();
			this.predictor = model.newPredictor();
		} catch (ModelNotFoundException | MalformedModelException | IOException e) {
			e.printStackTrace();
		}
	}
	
	public Detector() {
		this("craft");
	}

	// Add this getter method mainly for unit testing
	public Predictor<Image, BBox[]> getDetectorPredictor() {
		return this.predictor;
	}
	
	/**
	 * @param image
	 * Process the image Matrix and invoke internal implementation details
	 * to predict the bounding boxes that contains potential characters 
	 */
	public BBox[] detectBBoxes(String image_path) {
		Image img = null;
		try {
			img = ImageFactory.getInstance().fromFile(Paths.get(image_path));
		} catch (IOException e) { 
			e.printStackTrace();
		}
		img.getWrappedImage();
		BBox[] boxes = null;
		try {
			boxes = predictor.predict(img);
		} catch (TranslateException e) {
			e.printStackTrace();
		}
		return boxes;
	}
}
