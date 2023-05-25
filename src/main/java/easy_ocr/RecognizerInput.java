package easy_ocr;

import ai.djl.modality.cv.Image;

public class RecognizerInput {
	Image image;
	Boolean contrast;
	BBox bbox;
	
	public RecognizerInput(BBox box, Image img) {
		bbox = box;
		image = img;
		contrast = false;
	}
	
	public RecognizerInput() {
		this(new BBox(), null);
	}
}