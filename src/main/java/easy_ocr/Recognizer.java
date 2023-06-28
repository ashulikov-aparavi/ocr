package easy_ocr;

import java.util.concurrent.ConcurrentHashMap;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashMap;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

public class Recognizer {
	Predictor<RecognizerInput, RecognizerOutput> predictor;
	String currentModel;
	
	private static final HashMap<String, String> charSet = new HashMap<>();
	private static final String modelPath = "models/";

    static {
    	charSet.put("latin_gen2", " !\"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~ªÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿĀāĂăĄąĆćČčĎďĐđĒēĖėĘęĚěĞğĨĩĪīĮįİıĶķĹĺĻļĽľŁłŃńŅņŇňŒœŔŕŘřŚśŞşŠšŤťŨũŪūŮůŲųŸŹźŻżŽžƏƠơƯưȘșȚțə̇ḌḍḶḷṀṁṂṃṄṅṆṇṬṭẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ€");
    	charSet.put("cyrillic_gen2", "0123456789!\"#$%&\\'()*+,-./:;<=>?@[\\\\]^_`{|}~ €₽ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяЂђЃѓЄєІіЇїЈјЉљЊњЋћЌќЎўЏџҐґҒғҚқҮүҲҳҶҷӀӏӢӣӨөӮӯ");
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

	

	/***
	 * @param image
	 * @param bboxes
	 * @return list of predicted output words along with the bbox coordinates
	 * 
	 * This API is responsible for processing the image stream with the provided bounding boxes that
	 * generated from Detector model, crop and scale those boxes to a convenient shape and invoke 
	 * the recognizer model on each reformed bbox to get the set of predicted character set   
	 */

	public Prediction recognizeFromBbox(RecognizerInput inp) {
		Prediction recognition = new Prediction();
		try {
			recognition = new Prediction(inp.bbox);
			RecognizerOutput output = predictor.predict(inp);
			//System.out.println("Current score: " + output.score);
			if (output.score > 0.1) {
				recognition.word = output.ans;
			}
			else {
				inp.contrast = true;
				RecognizerOutput new_output = predictor.predict(inp);
				//System.out.println("Score after contrast: " + output.score);
				if (new_output.score > 0.1) {
					recognition.word = new_output.ans;
				}
				else {
					inp.preprocessing = true;
					RecognizerOutput last_output = predictor.predict(inp);
					if ((output.score > new_output.score) & (output.score > last_output.score)) {
						recognition.word = output.ans;
					}
					else if ((new_output.score > output.score) & (new_output.score > last_output.score)) {
						recognition.word = new_output.ans;
					}
					else {
						recognition.word = last_output.ans;
					}
				}
			}
		} catch (TranslateException e) {
			e.printStackTrace();
		}
		return recognition;
	}
}
