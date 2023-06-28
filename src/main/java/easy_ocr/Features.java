package easy_ocr;

import ai.djl.ndarray.NDArray;

public class Features {
	NDArray normal;
	NDArray contrast;
	NDArray preprocessing;
	BBox bbox;
	
	public Features(BBox box) {
		normal = null;
		contrast = null;
		preprocessing = null;
		bbox = box;
	}
}
