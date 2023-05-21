package easy_ocr;

public class Prediction {
	String word;
	BBox bbox;
	
	public Prediction() {
		word = "";
	}
	
	public Prediction(BBox box) {
		word = "";
		bbox = box;
	}

	// API for accessing the private member 'bbox' & 'word'
	public BBox getBBox() {
		return this.bbox;
	}

	public String getWord() {
		return this.word;
	}
	
	public double getCustomProperty(int num) {
		if (num == 0) {
			return this.bbox.points.get(0).y;
		}
		else {
			return this.bbox.points.get(0).x;
		}
    }
}