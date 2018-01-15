package gttt.cds.test.facerecognition;

import java.io.File;
//import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
//import org.opencv.contrib.FaceRecognizer;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.test.facerecognition.R;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.KeyEvent;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;


public class FaceRecognitionActivity extends Activity implements CvCameraViewListener2 {

    private static final String    TAG                 = "OCVSample::Activity";
    private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;
    
    public static final int TRAINING = 0;
    public static final int SEARCHING = 1;
    public static final int IDLE = 2;
    
    private static final int frontCam =1;
    private static final int backCam =2;
    	    		
    
    private int mState = IDLE;
//    private int countTrain=0;
    
//    private MenuItem               mItemFace50;
//    private MenuItem               mItemFace40;
//    private MenuItem               mItemFace30;
//    private MenuItem               mItemFace20;
//    private MenuItem               mItemType;
//    
    private MenuItem               nBackCam;
    private MenuItem               mFrontCam;
    private MenuItem               mEigen;
    

    private Mat                    mRgba;
    private Mat                    mGray;
    private File                   mCascadeFile;
    private CascadeClassifier      mJavaDetector;
 //   private DetectionBasedTracker  mNativeDetector;

    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;

    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;
    private int mAccuracy =999;
    
    String mPath="";

    private FaceRecognitionView mFaceRecognitionView;
    private int mChooseCamera = backCam;
    
    EditText mNameInput;
    ImageView mFoundFacesPrevView;
    Bitmap mBitmap;
    Handler mHandler;
  
    PersonRecognizer mPersonRecognizer;
    ToggleButton mToggleButtonRec, mToggleButtonTrain, mButtonSearch;
    Button mButtonGallery;
    ImageButton mSelectButtonCamera;
    
    TextView mTextState, mTextAccuracy, mTextResult;
    com.googlecode.javacv.cpp.opencv_contrib.FaceRecognizer faceRecognizer;
   
    
    static final long MAXIMG = 10;
    
    ArrayList<Mat> alimgs = new ArrayList<Mat>();

    int[] labels = new int[(int)MAXIMG];
    int mCountImages =0;
    
    Labels labelsFile;
	static {
		OpenCVLoader.initDebug();            
    	System.loadLibrary("opencv_java");	       	 
	}
    
    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    mPersonRecognizer =new PersonRecognizer(mPath, getApplicationContext());
                    mPersonRecognizer.load();
                    
                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else {
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());
                        }
                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mFaceRecognitionView.enableView();
                    break;
                }
                default: {
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };

    public FaceRecognitionActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "onCreate()");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        mPath = getFilesDir() + "/gttt_facerecognition/";
        mHandler = new H();

        initView();

        mNameInput.setVisibility(View.INVISIBLE);
        mTextResult.setVisibility(View.INVISIBLE);
        mToggleButtonRec.setVisibility(View.INVISIBLE);

        boolean mkDirSuccess = (new File(mPath)).mkdirs();
        if (!mkDirSuccess) {
            String s = getResources().getString(R.string.SMkdirError);
        	Log.e("Error", s);
            Toast.makeText(getApplicationContext(), s, Toast.LENGTH_LONG).show();
        }
    }

    void initView() {
        mFaceRecognitionView = (FaceRecognitionView) findViewById(R.id.tutorial3_activity_java_surface_view);
        mFaceRecognitionView.setCvCameraViewListener(this);

        mFoundFacesPrevView = (ImageView)findViewById(R.id.img_found_faces);
        mTextResult = (TextView) findViewById(R.id.txt_found_face);
        mTextAccuracy = (TextView) findViewById(R.id.txt_accuracy);
        mTextState = (TextView)findViewById(R.id.txt_mode);

        mNameInput = (EditText)findViewById(R.id.editText1);
        mButtonGallery = (Button)findViewById(R.id.btn_gallery);
        mToggleButtonRec = (ToggleButton)findViewById(R.id.toggleButtonRec);
        mButtonSearch = (ToggleButton)findViewById(R.id.buttonBuscar);
        mToggleButtonTrain = (ToggleButton)findViewById(R.id.toggleButton1);
        mSelectButtonCamera =(ImageButton)findViewById(R.id.img_cam);

        setListeners();
    }

    void setListeners() {
        mButtonGallery.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
                Intent i = new Intent(FaceRecognitionActivity.this, ImageGallery.class);
                i.putExtra("path", mPath);
                startActivity(i);
            };
        });

        mNameInput.setOnKeyListener(new View.OnKeyListener() {
            public boolean onKey(View v, int keyCode, KeyEvent event) {
                if ((mNameInput.getText().toString().length()>0)&&(mToggleButtonTrain.isChecked()))
                    mToggleButtonRec.setVisibility(View.VISIBLE);
                else
                    mToggleButtonRec.setVisibility(View.INVISIBLE);
                return false;
            }
        });

        mToggleButtonTrain.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                if (mToggleButtonTrain.isChecked()) {
                    mTextState.setText(getResources().getString(R.string.SEnter));
                    mButtonSearch.setVisibility(View.INVISIBLE);
                    mButtonGallery.setVisibility(View.INVISIBLE);
                    mTextResult.setVisibility(View.VISIBLE);
                    mNameInput.setVisibility(View.VISIBLE);
                    mTextResult.setText(getResources().getString(R.string.SFaceName));
                    if (mNameInput.getText().toString().length() > 0)
                        mToggleButtonRec.setVisibility(View.VISIBLE);

                } else {
                    mTextState.setText(R.string.Straininig);
                    mTextResult.setText("");
                    mNameInput.setVisibility(View.INVISIBLE);

                    mButtonSearch.setVisibility(View.VISIBLE);
                    mButtonGallery.setVisibility(View.VISIBLE);
                    mToggleButtonRec.setVisibility(View.INVISIBLE);
                    mNameInput.setVisibility(View.INVISIBLE);

                    mPersonRecognizer.train();
                    mTextState.setText(getResources().getString(R.string.SIdle));
                }
            }

        });

        mToggleButtonRec.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                recButtonOnclick();
            }
        });

        mSelectButtonCamera.setOnClickListener(new View.OnClickListener() {

            public void onClick(View v) {
                if (mChooseCamera==frontCam) {
                    mChooseCamera=backCam;
                    mFaceRecognitionView.setCamBack();
                } else {
                    mChooseCamera=frontCam;
                    mFaceRecognitionView.setCamFront();
                }
            }
        });

        mButtonSearch.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                if (mButtonSearch.isChecked()) {
                    if (!mPersonRecognizer.canPredict()) {
                        mButtonSearch.setChecked(false);
                        Toast.makeText(getApplicationContext(), getResources().getString(R.string.SCanntoPredic), Toast.LENGTH_LONG).show();
                        return;
                    }
                    mState = SEARCHING;
                    mTextState.setText(getResources().getString(R.string.SSearching));
                    mToggleButtonRec.setVisibility(View.INVISIBLE);
                    mToggleButtonTrain.setVisibility(View.INVISIBLE);
                    mNameInput.setVisibility(View.INVISIBLE);
                    mTextResult.setVisibility(View.VISIBLE);
                } else {
                    mState = IDLE;
                    mTextState.setText(getResources().getString(R.string.SIdle));
                    mToggleButtonRec.setVisibility(View.INVISIBLE);
                    mToggleButtonTrain.setVisibility(View.VISIBLE);
                    mNameInput.setVisibility(View.INVISIBLE);
                    mTextResult.setVisibility(View.INVISIBLE);
                }
            }
        });
    }
    
    void recButtonOnclick() {
    	if (mToggleButtonRec.isChecked()) {
            mState = TRAINING;
        } else {
         // train();
          //mPersonRecognizer.train();
          mCountImages = 0;
          mState =IDLE;
        }
    }
    
    @Override
    public void onPause() {
        super.onPause();
        if (mFaceRecognitionView != null)
            mFaceRecognitionView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
       // OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        mFaceRecognitionView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    // called upon camera rendering
    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Log.e(TAG, "onCameraFrame()");
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
          //  mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }

        MatOfRect faces = new MatOfRect();

        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                // detect faces
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        } else if (mDetectorType == NATIVE_DETECTOR) {
//            if (mNativeDetector != null)
//                mNativeDetector.detect(mGray, faces);
        } else {
            Log.e(TAG, "Detection method is not selected!");
            return mRgba;
        }

        Rect[] facesArray = faces.toArray();
        if ((facesArray.length==1)
                && (mState == TRAINING)
                && (mCountImages < MAXIMG)
                && (!mNameInput.getText().toString().isEmpty())) {
            Mat m = new Mat();
            Rect r = facesArray[0];
            m = mRgba.submat(r);
            mBitmap = Bitmap.createBitmap(m.width(),m.height(), Bitmap.Config.ARGB_8888);

            Utils.matToBitmap(m, mBitmap);
           // SaveBmp(mBitmap,"/sdcard/db/I("+countTrain+")"+mCountImages+".jpg");

            mHandler.obtainMessage(H.SET_FOUND_FACES, mBitmap).sendToTarget();
            if (mCountImages < MAXIMG) {
                mPersonRecognizer.addData(m, mNameInput.getText().toString());
                mCountImages++;
            }

        } else if ((facesArray.length>0)&& (mState == SEARCHING)) {
            Mat m = new Mat();
            m = mGray.submat(facesArray[0]);
            mBitmap = Bitmap.createBitmap(m.width(),m.height(), Bitmap.Config.ARGB_8888);

            Utils.matToBitmap(m, mBitmap);
            mHandler.obtainMessage(H.SET_FOUND_FACES, mBitmap).sendToTarget();

            // predict faces with detected face area
            String predictResult = mPersonRecognizer.predict(m);
            mHandler.obtainMessage(H.SET_TEXT_RESULT, "Result: " + predictResult).sendToTarget();

            mAccuracy = mPersonRecognizer.getProb();
            String accuracy = "" + mAccuracy;
            mHandler.obtainMessage(H.SET_TEXT_ACCURACY, "Accuracy: " + accuracy).sendToTarget();
          }
//        for (int i = 0; i < facesArray.length; i++)
//            Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);

        return mRgba;
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "onCreateOptionsMenu()");
        if (mFaceRecognitionView.numberCameras()>1) {
        nBackCam = menu.add(getResources().getString(R.string.SFrontCamera));
        mFrontCam = menu.add(getResources().getString(R.string.SBackCamera));
//        mEigen = menu.addData("EigenFaces");
//        mLBPH.setChecked(true);
        } else {
            mSelectButtonCamera.setVisibility(View.INVISIBLE);
        }
        //mFaceRecognitionView.setAutofocus();
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "onOptionsItemSelected(): selected item: " + item);
        nBackCam.setChecked(false);
        mFrontCam.setChecked(false);
      //  mEigen.setChecked(false);
        if (item == nBackCam)
        {
        	mFaceRecognitionView.setCamFront();
        	mChooseCamera=frontCam;
        }
        	//mPersonRecognizer.changeRecognizer(0);
        else if (item==mFrontCam)
        {
        	mChooseCamera=backCam;
        	mFaceRecognitionView.setCamBack();
        	
        }
       
        item.setChecked(true);
       
        return true;
    }

    final class H extends Handler {
        static final int SET_FOUND_FACES = 0;
        static final int SET_TEXT_RESULT = 1;
        static final int SET_TEXT_ACCURACY = 2;
        @Override
        public void handleMessage(Message msg) {
            switch(msg.what) {
                case SET_FOUND_FACES: {
                    Canvas canvas = new Canvas();
                    Bitmap b = (Bitmap)msg.obj;
                    canvas.setBitmap(b);
                    mFoundFacesPrevView.setImageBitmap(b);
                    if (mCountImages >= MAXIMG - 1) {
                        mToggleButtonRec.setChecked(false);
                        recButtonOnclick();
                    }
                    break;
                }
                case SET_TEXT_RESULT: {
                    mTextResult.setText(msg.obj.toString());
                    break;
                }
                case SET_TEXT_ACCURACY: {
                    mTextAccuracy.setText(msg.obj.toString());
                    break;
                }
            }
        }
    }
    

}
