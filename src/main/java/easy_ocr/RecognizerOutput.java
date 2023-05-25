package easy_ocr;

public class RecognizerOutput {
	String ans;
	double score;
	
	public RecognizerOutput(double g, String a) {
		ans = a;
		score = g;
	}
	
	public RecognizerOutput() {
		this(0.0, "");
	}
}
