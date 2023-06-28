package easy_ocr;

import java.util.Arrays;
import java.util.List;

import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.*;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.*;

import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;


public class ScriptIdentificationTranslator implements NoBatchifyTranslator<RecognizerInput, Classifications>{
	private int width;

    /**
     * Creates the {@link DetectorTranslator} instance.
     *
     * @param arguments the arguments for the translator
     */
    public ScriptIdentificationTranslator() {
    }

    /** {@inheritDoc} */
    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        NDArray probabilities = list.singletonOrThrow().softmax(0);
        List<String> classNames = IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
        return new Classifications(classNames, probabilities);
    }
    
    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, RecognizerInput input) {
    	NDManager manager = ctx.getNDManager();
        NDArray img = input.image.toNDArray(manager, Image.Flag.GRAYSCALE);
        width = input.image.getWidth();
        img = NDImageUtils.toTensor(img);
        img = img.expandDims(0);
        return new NDList(img);
    }
}