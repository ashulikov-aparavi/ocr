package easy_ocr;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

public class ScriptIdentificator {
	Predictor<RecognizerInput, Classifications> predictor;
	String currentModel;
	
	private static final HashMap<String, String> charSet = new HashMap<>();
	private static final String modelPath = "models/script_ident";
    
    public ScriptIdentificator(String modelType) {
    	Criteria<RecognizerInput, Classifications> criteria = Criteria.builder()
				.setTypes(RecognizerInput.class, Classifications.class)
		        .optModelPath(Paths.get(modelPath))
		        .optTranslator(new ScriptIdentificationTranslator())
		        .optProgress(new ProgressBar()).build();

		try {
			ZooModel<RecognizerInput, Classifications> model = criteria.loadModel();
			this.predictor = model.newPredictor();
		} catch (ModelNotFoundException | MalformedModelException |IOException e) {
			e.printStackTrace();
		}
	}
    
	public ScriptIdentificator() {
		this("latin_gen2");
	}
	
	public Predictor<RecognizerInput, Classifications> getDetectorPredictor() {
		return this.predictor;
	}

	// For checking which char set we support during training the recognizer model
	public String supportedChars() {
		return charSet.get(currentModel);
	}

	

	/***
	 * @param image_list
	 * @return list of predicted script numerical representations
	 * 
	 * This API is responsible for processing the image stream with the provided bounding boxes that
	 * generated from Detector model, crop and scale those boxes to a convenient shape and invoke 
	 * the recognizer model on each reformed bbox to get the set of predicted character set   
	 */

	public int[] getScript(ArrayList<RecognizerInput> image_list) {
		int[] scripts = new int[image_list.size()];
		for (int i = 0; i < image_list.size(); ++i) {
			try {
				RecognizerInput inp = image_list.get(i);
				scripts[i] = Integer.parseInt(predictor.predict(inp).best().getClassName());
			} catch (TranslateException e) {
				e.printStackTrace();
			}
		}
		return scripts;
	}
}
