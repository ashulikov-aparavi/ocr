package easy_ocr;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;

public class BBox {
	List<Point> points;
	double width;
	double height;
	double maxX;
	double maxY;
	double minX;
	double minY;

	private static final int BBOX_COORDINATES = 4;

	// Added this to avoid exception during BBox creation with default coordinates
	public BBox() {
		
	}

	public BBox(double left, double right, double top, double bottom) {
		points = new ArrayList<>();
		width = Math.sqrt(Math.pow(left - right, 2));
		height = Math.sqrt(Math.pow(top - bottom, 2));
		minX = left;
		maxX = right;
		minY = top;
		maxY = bottom;
		points.add(new Point(left, top));
		points.add(new Point(right, top));
		points.add(new Point(right, bottom));
		points.add(new Point(left, bottom));
	}
	
	public BBox(Mat box) {
		new BBox(box, 1.0);
	}
	
	public BBox(Mat box, double scale) {
		double mi = 100000000;
		minX = 100000000;
		minY = 100000000;
		maxX = 0;
		maxY = 0;
		int ind = 0;
		for (int i = 0; i < 4; ++i) {
			minX = Math.min(box.get(i, 0)[0]*scale, minX);
			maxX = Math.max(box.get(i, 0)[0]*scale, maxX);
			minY = Math.min(box.get(i, 1)[0]*scale, minY);
			maxY = Math.max(box.get(i, 1)[0]*scale, maxY);
			if (box.get(i, 0)[0] + box.get(i, 1)[0] < mi) {
				mi = box.get(i, 0)[0] + box.get(i, 1)[0];
				ind = i;
			}
		}
		points = new ArrayList<>();
		for (int i = ind; i < ind + BBOX_COORDINATES; ++i) {
			points.add(new Point(box.get(i % BBOX_COORDINATES, 0)[0]*scale, box.get(i % BBOX_COORDINATES, 1)[0]*scale));
		}
		this.width = (Math.sqrt(Math.pow(points.get(0).x - points.get(1).x, 2) + 
	   		   	   	   Math.pow(points.get(0).y - points.get(1).y, 2)) +
			 Math.sqrt(Math.pow(points.get(2).x - points.get(3).x, 2) + 
					   Math.pow(points.get(2).y - points.get(3).y, 2))) / 2;
		this.height = (Math.sqrt(Math.pow(points.get(0).x - points.get(3).x, 2) + 
		   		   	   Math.pow(points.get(0).y - points.get(3).y, 2)) +
			 Math.sqrt(Math.pow(points.get(2).x - points.get(1).x, 2) + 
					   Math.pow(points.get(2).y - points.get(1).y, 2))) / 2;
	}
	
	public BBox(List<Point> box) {
		points = box;
		minX = 100000000;
		minY = 100000000;
		maxX = 0;
		maxY = 0;
		for (int i = 0; i < 4; ++i) {
			minX = Math.min(box.get(i).x, minX);
			maxX = Math.max(box.get(i).x, maxX);
			minY = Math.min(box.get(i).y, minY);
			maxY = Math.max(box.get(i).y, maxY);
		}
		this.width = (Math.sqrt(Math.pow(points.get(0).x - points.get(1).x, 2) + 
	   		   	   	   Math.pow(points.get(0).y - points.get(1).y, 2)) +
			 Math.sqrt(Math.pow(points.get(2).x - points.get(3).x, 2) + 
					   Math.pow(points.get(2).y - points.get(3).y, 2))) / 2;
		this.height = (Math.sqrt(Math.pow(points.get(0).x - points.get(3).x, 2) + 
		   		   	   Math.pow(points.get(0).y - points.get(3).y, 2)) +
			 Math.sqrt(Math.pow(points.get(2).x - points.get(1).x, 2) + 
					   Math.pow(points.get(2).y - points.get(1).y, 2))) / 2;
	}
	
	public Mat getMat() {
		MatOfPoint mPoints = new MatOfPoint();
		mPoints.fromList(points);
		Mat ans = (Mat) mPoints;
		ans.convertTo(ans, CvType.CV_32F);
		return (Mat) mPoints;
	}

	// Set BBox with custom Coordinates
	// Mainly added for unit test
	public void setPoints(List<Point> Points) {
		this.points = Points;
	}
}
