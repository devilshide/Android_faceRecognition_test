package gttt.cds.test.facerecognition;

import static  com.googlecode.javacv.cpp.opencv_highgui.*;
import static  com.googlecode.javacv.cpp.opencv_core.*;

import static  com.googlecode.javacv.cpp.opencv_imgproc.*;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FilenameFilter;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.test.facerecognition.R;

import com.googlecode.javacv.cpp.opencv_imgproc;
import com.googlecode.javacv.cpp.opencv_contrib.FaceRecognizer;
import com.googlecode.javacv.cpp.opencv_core.IplImage;
import com.googlecode.javacv.cpp.opencv_core.MatVector;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.widget.Toast;

public class PersonRecognizer {
    final static String TAG = "PersonRecognizer";
    Context mContext;
	FaceRecognizer mFaceRecognizer;
	String mPath;
	int mCount =0;
	Labels mLabelsFile;

    static  final int WIDTH= 128;
    static  final int HEIGHT= 128;;
    private int mProb=999;
	
	 
    PersonRecognizer(String path, Context context) {
        mContext = context;
    	// path=Environment.getExternalStorageDirectory()+"/facerecog/faces/";
		mPath = path;
		mLabelsFile = new Labels(mPath);
        mFaceRecognizer =  com.googlecode.javacv.cpp.opencv_contrib.createLBPHFaceRecognizer(2,8,8,8,200);
    }

	void addData(Mat m, String name) {
		Bitmap bmp= Bitmap.createBitmap(m.width(), m.height(), Bitmap.Config.ARGB_8888);
		 
		Utils.matToBitmap(m,bmp);
		bmp= Bitmap.createScaledBitmap(bmp, WIDTH, HEIGHT, false);

		try {
			FileOutputStream f = new FileOutputStream(mPath + name + "-" + mCount + ".jpg",true);
			mCount++;
			bmp.compress(Bitmap.CompressFormat.JPEG, 100, f);
			f.close();

		} catch (Exception e) {
			Log.e("error",e.getCause()+" "+e.getMessage());
			e.printStackTrace();
			
		}
	}
	
	public boolean train() {
        Toast.makeText(mContext, mContext.getResources().getString(R.string.Straininig), Toast.LENGTH_LONG).show();

		File root = new File(mPath);
        FilenameFilter pngFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.toLowerCase().endsWith(".jpg");
        	};
        };

        File[] imageFiles = root.listFiles(pngFilter);

        MatVector images = new MatVector(imageFiles.length);

        int[] labels = new int[imageFiles.length];

        int counter = 0;
        int label;

        IplImage img = null;
        IplImage grayImg = null;
        int i1 = mPath.length();
        for (File image : imageFiles) {
        	String imagePath = image.getAbsolutePath();
            img = cvLoadImage(imagePath);
            
            if (img == null) {
                Log.e("Error", "Error cVLoadImage");
                continue;
            }
            Log.i(TAG, "train(): img path = " + imagePath);
            
            int i2 = imagePath.lastIndexOf("-");
            int i3 = imagePath.lastIndexOf(".");
            int icount = Integer.parseInt(imagePath.substring(i2+1, i3));
            if (mCount < icount) mCount++;
            
            String description=imagePath.substring(i1,i2);
            
            if (mLabelsFile.get(description)<0)
            	mLabelsFile.add(description, mLabelsFile.max()+1);
            
            label = mLabelsFile.get(description);

            grayImg = IplImage.create(img.width(), img.height(), IPL_DEPTH_8U, 1);

            cvCvtColor(img, grayImg, CV_BGR2GRAY);

            images.put(counter, grayImg);

            labels[counter] = label;

            counter++;
        }
        if (counter>0)
        	if (mLabelsFile.max()>1)
        		mFaceRecognizer.train(images, labels);
        mLabelsFile.Save();
		return true;
	}
	
	public boolean canPredict() {
		if (mLabelsFile.max() > 1)
			return true;
		else
			return false;
	}
	
	public String predict(Mat m) {
		if (!canPredict())
			return "";
		int n[] = new int[1];
		double p[] = new double[1];
		IplImage ipl = MatToIplImage(m,WIDTH, HEIGHT);

		mFaceRecognizer.predict(ipl, n, p);
		
		if (n[0]!=-1)
		 mProb=(int)p[0];
		else
			mProb=-1;
	//	if ((n[0] != -1)&&(p[0]<95))
		if (n[0] != -1)
			return mLabelsFile.get(n[0]);
		else
			return "Unkown";
	}

    IplImage MatToIplImage(Mat m,int width,int heigth) {
       Bitmap bmp=Bitmap.createBitmap(m.width(), m.height(), Bitmap.Config.ARGB_8888);
       Utils.matToBitmap(m, bmp);
       return BitmapToIplImage(bmp,width, heigth);
    }

	IplImage BitmapToIplImage(Bitmap bmp, int width, int height) {
		if ((width != -1) || (height != -1)) {
			Bitmap bmp2 = Bitmap.createScaledBitmap(bmp, width, height, false);
			bmp = bmp2;
		}

		IplImage image = IplImage.create(bmp.getWidth(), bmp.getHeight(),
				IPL_DEPTH_8U, 4);

		bmp.copyPixelsToBuffer(image.getByteBuffer());
		
		IplImage grayImg = IplImage.create(image.width(), image.height(),
				IPL_DEPTH_8U, 1);

		cvCvtColor(image, grayImg, opencv_imgproc.CV_BGR2GRAY);

		return grayImg;
	}


	public void load() {
		train();
	}

    public int getProb() {
        return mProb;
    }

}
