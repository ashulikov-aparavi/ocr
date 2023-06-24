package easy_ocr;

import java.util.Arrays;
import java.util.Map;

import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.*;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.*;

import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class RecognizerTranslator implements NoBatchifyTranslator<RecognizerInput, RecognizerOutput>{
	private int width;
	private int height;
	private String charSet;
	private static final float val_diff = 0.03f;

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
        NDArray prediction_numbers = prediction.get(new NDIndex(":, :, 17:26"));
        NDArray prediction_letters = prediction.get(new NDIndex(":, :, 34:"));
        NDArray values = prediction.max(new int[]{2});
        NDArray preds_index = prediction.argMax(2);
        NDArray number_values = prediction_numbers.max(new int[]{2});
        NDArray letter_values = prediction_letters.max(new int[]{2});
        NDArray number_index = prediction_numbers.argMax(2);
        NDArray letter_index = prediction_letters.argMax(2);
        String text = "";
        String s = "0123456789";
        Boolean number = false;
        if (preds_index.getLong(0, 0) != 0) {
        	char curChar = charSet.charAt((int)preds_index.getLong(0, 0)-1);
        	text += curChar;
        	number = s.indexOf(curChar) > -1;
    	}
        for (int i = 1; i < length; ++i) {
        	long cur = preds_index.getLong(0, i);
        	if ((cur != 0) && (cur != preds_index.getLong(0, i-1))) {
            	char curChar = charSet.charAt((int)preds_index.getLong(0, i)-1);
            	double curVal = values.getFloat(0, i);
            	if (number ^ (s.indexOf(curChar) > -1)) {
            		if (number && (curVal - number_values.getFloat(0, i) < val_diff)) {
            			curChar = charSet.charAt((int)number_index.getLong(0, i));
            		}
            		if (!number && (curVal - letter_values.getFloat(0, i) < val_diff)) {
            			curChar = charSet.charAt((int)letter_index.getLong(0, i));
            		}
            	}
            	text += curChar;
            	number = s.indexOf(curChar) > -1;
        	}
        }
        text = postprocessText(text);
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
    
    public static String postprocessText(String inputString) {
    	String withDates = listPotentialDates(inputString);
    	String withEmails = listPotentialEmails(withDates);
    	String withIBANS = listPotentialIBANS(withEmails);
        return withIBANS;
    }
    
    public static String listPotentialIBANS(String inputLine) {
        String regex = "IBAN.+";
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(inputLine);
        while (matcher.find()) {
            String iban = matcher.group();
            inputLine = inputLine.replace(iban, refineIBAN(iban));
        }
        return inputLine;
    }
    
    public static String refineIBAN(String ibanStr) {
    	ibanStr = ibanStr.replaceAll("[,\\s\\\\/]", "");

        // Regex pattern to match potential email addresses
        Pattern pattern = Pattern.compile("IBAN[:|](\\S+)");

        // Matcher to find and replace the patterns
        Matcher matcher = pattern.matcher(ibanStr);

        // StringBuilder to store the processed string
        StringBuilder outputDate = new StringBuilder();

        // Iterate over the matches and replace them with the valid email addresses
        while (matcher.find()) {
            String ibanNumber = matcher.group(1);
            // Check if the domain ends with a period
            // Build the email address with or without the period based on the condition
            matcher.appendReplacement(outputDate, ibanNumber.toUpperCase());
        }
        matcher.appendTail(outputDate);
        String processedString = outputDate.toString();
        return processedString;
    }
    
    public static String listPotentialDates(String inputLine) {
        String regex = "\\S+\\s*[/|-|.|,]\\s*\\S+\\s*[/|-|.|,]\\s*\\S+";
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(inputLine);
        while (matcher.find()) {
            String date = matcher.group();
            inputLine = inputLine.replace(date, refineDate(date));
        }
        return inputLine;
    }
    
    public static String refineDate(String dateStr) {
    	dateStr = dateStr.replaceAll("[,\\s]", "");

        // Regex pattern to match potential email addresses
        Pattern pattern = Pattern.compile("(\\d{1,4})(/|-|.|,)(\\d{1,4})(/|-|.|,)(\\d{1,4})");

        // Matcher to find and replace the patterns
        Matcher matcher = pattern.matcher(dateStr);

        // StringBuilder to store the processed string
        StringBuilder outputDate = new StringBuilder();

        // Iterate over the matches and replace them with the valid email addresses
        while (matcher.find()) {
            String day = matcher.group(1);
            String separator = matcher.group(2);
            if (separator == ",") {
            	separator = ".";
            }
            String month = matcher.group(3);
            String year = matcher.group(5);
            // Check if the domain ends with a period
            // Build the email address with or without the period based on the condition
            String date = day + separator + month + separator + year;

            matcher.appendReplacement(outputDate, date);
        }
        matcher.appendTail(outputDate);
        String processedString = outputDate.toString();
        return processedString;
    }
    

    /**
     * Check for potential email information
     * in the input string
     * 
     * @return
     */
    public static String listPotentialEmails(String inputLine) {
        String emailPattern = "\\S+@.*?(com|de|net|org|eu|gov|info)";
        Matcher emailMatcher = Pattern.compile(emailPattern).matcher(inputLine);

        while (emailMatcher.find()) {
            String email = emailMatcher.group();
            inputLine = inputLine.replace(email, refineEmail(email));
        }

        return inputLine;
    }
    
    public static String refineEmail(String emailStr) {
        emailStr = emailStr.replaceAll("[,\\s\\\\/]", "");

        // Regex pattern to match potential email addresses
        Pattern pattern = Pattern.compile("(\\S+)@(\\S+)(com|de|net|org|eu|gov|info)");

        // Matcher to find and replace the patterns
        Matcher matcher = pattern.matcher(emailStr);

        // StringBuilder to store the processed string
        StringBuilder outputEmail = new StringBuilder();

        // Iterate over the matches and replace them with the valid email addresses
        while (matcher.find()) {
            String username = matcher.group(1);
            String domain = matcher.group(2);
            String extension = matcher.group(3);
            // Check if the domain ends with a period
            boolean hasPeriod = domain.endsWith(".");
            // Build the email address with or without the period based on the condition
            String emailAddress;
            if (hasPeriod) {
                emailAddress = username + "@" + domain + extension;
            } else {
                emailAddress = username + "@" + domain + "." + extension;
            }

            matcher.appendReplacement(outputEmail, emailAddress);
        }
        matcher.appendTail(outputEmail);
        String processedString = outputEmail.toString();
        return processedString;

    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, RecognizerInput input) {
    	NDManager manager = ctx.getNDManager();
        NDArray img = input.image.toNDArray(manager, Image.Flag.GRAYSCALE);
        img = img.toType(DataType.INT32, true);
        width = input.image.getWidth();
        height = input.image.getHeight();
        if (input.contrast) {
        	img = contrastAdd(img);
        }
        img = img.toType(DataType.INT32, true);
        if (input.preprocessing) {
        	img = preprocess(img);
        }
        NDArray out = manager.zeros(new Shape(1, (int)(width/10.0 + 1)));
        img = NDImageUtils.toTensor(img);
        img = img.expandDims(0);
        return new NDList(img, out);
    }
    
    private NDArray preprocess(NDArray image) {
        Mat matImg = new Mat(height, width, CvType.CV_8UC1);
        for (int i=0; i<height; i++) {
            for(int j=0; j<width; j++) {
            	matImg.put(i, j, image.getInt(i, j, 0));
            }
        }
        Imgproc.equalizeHist(matImg, matImg);
        Imgproc.GaussianBlur(matImg, matImg, new Size(5, 5), 0);
        Imgproc.threshold(matImg, matImg, 0, 255, Imgproc.THRESH_OTSU);
        for (int i=0; i<height; i++) {
            for(int j=0; j<width; j++) {
            	image.set(new NDIndex(i, j), matImg.get(i, j)[0]);
            }
        }
        return image;
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