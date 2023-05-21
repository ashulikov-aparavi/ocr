package easy_ocr;

import java.util.Map;

import ai.djl.modality.cv.*;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.*;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.*;


public class RecognizerTranslator implements NoBatchifyTranslator<Image, String>{
	private int width;
	private String charSet;

    /**
     * Creates the {@link DetectorTranslator} instance.
     *
     * @param arguments the arguments for the translator
     */
    public RecognizerTranslator(Map<String, ?> arguments) {
    	charSet = ArgumentsUtil.stringValue(arguments, "charSet");
    }

    /** {@inheritDoc} */
    @Override
    public String processOutput(TranslatorContext ctx, NDList list) {
        NDArray prediction = list.singletonOrThrow();
        int length = (int) prediction.size(1);

		System.out.println("Current image path: " + length);
        prediction = prediction.softmax(2);
        NDArray predNorm = prediction.sum(new int[] {2});
        prediction = prediction.div(predNorm.expandDims(-1));
        NDArray preds_index = prediction.argMax(2);
        String text = "";
        for (int i = 0; i < length; ++i) {
        	if (preds_index.getLong(0, i) != 0) {
        		text += charSet.charAt((int)preds_index.getLong(0, i)-1);
        	}
        }
        return text;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
    	NDManager manager = ctx.getNDManager();
        NDArray img = input.toNDArray(manager, Image.Flag.GRAYSCALE);
        width = input.getWidth();
        NDArray out = manager.zeros(new Shape(1, (int)(width/10.0)));
        img = NDImageUtils.toTensor(img);
        img = img.expandDims(0);
        return new NDList(img, out);
    }
}