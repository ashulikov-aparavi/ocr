package easy_ocr;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Map;

import org.apache.commons.math3.stat.descriptive.rank.Percentile;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.*;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.opencv.OpenCVImageFactory;
import ai.djl.translate.*;


public class RecognizerTranslator implements NoBatchifyTranslator<RecognizerInput, RecognizerOutput>{
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
    public RecognizerOutput processOutput(TranslatorContext ctx, NDList list) {
        NDArray prediction = list.singletonOrThrow();
        int length = (int) prediction.size(1);
        prediction = prediction.softmax(2);
        NDArray predNorm = prediction.sum(new int[] {2});
        prediction = prediction.div(predNorm.expandDims(-1));
        NDArray preds_index = prediction.argMax(2);
        String text = "";
        if (preds_index.getLong(0, 0) != 0) {
    		text += charSet.charAt((int)preds_index.getLong(0, 0)-1);
    	}
        for (int i = 1; i < length; ++i) {
        	long cur = preds_index.getLong(0, i);
        	if ((cur != 0) && (cur != preds_index.getLong(0, i-1))) {
        		text += charSet.charAt((int)preds_index.getLong(0, i)-1);
        	}
        }
        NDArray values = prediction.max(new int[]{2});
        NDArray indices = prediction.argMax(2);
        int cnt = 0;
        double prod = 1.0;
        for (int i = 0; i < length; ++i) {
        	if (indices.getLong(0, i) != 0) {
                cnt += 1;
                prod *= values.getFloat(0, i);
        	}
        }
        if (cnt > 0) {
            return new RecognizerOutput(Math.pow(prod, 2/Math.sqrt((double)cnt)), text);
        }
        return new RecognizerOutput(0, text);
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, RecognizerInput input) {
    	NDManager manager = ctx.getNDManager();
        NDArray img = input.image.toNDArray(manager, Image.Flag.GRAYSCALE);
        img = img.toType(DataType.INT32, true);
        if (input.contrast) {
        	img = contrastAdd(img);
        }
        width = input.image.getWidth();
        NDArray out = manager.zeros(new Shape(1, (int)(width/10.0 + 1)));
        img = NDImageUtils.toTensor(img);
        img = img.expandDims(0);
        return new NDList(img, out);
    }
    
    private NDArray contrastAdd(NDArray image) {
        double[] img = Arrays.stream(image.toIntArray()).asDoubleStream().toArray();
        Percentile perc = new Percentile();
        double high = perc.evaluate(img, 90.0);
        double low = perc.evaluate(img, 10.0);
        if ((high-low)/Math.max(10, high+low) < 0.5) {
        	double ratio = 200.f/Math.max(10, high-low);
    		image = image.sub((int)low - 25);
        	image = image.mul(ratio);
        	image = image.clip(0, 255);
        }
        return image;
	}
}