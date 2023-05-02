package org.pytorch.demo.speechrecognition;

import android.content.Context;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;

import org.pytorch.LiteModuleLoader;


public class MainActivity extends AppCompatActivity implements Runnable {
    private static final String TAG = MainActivity.class.getName();

    private Module module;
    private TextView mTextView;
    private Button mButton;

    private final static int REQUEST_RECORD_AUDIO = 13;
    private final static int AUDIO_LEN_IN_SECOND = 1;
    private final static int SAMPLE_RATE = 8000;
    private final static int RECORDING_LENGTH = SAMPLE_RATE * AUDIO_LEN_IN_SECOND;

    private final static String LOG_TAG = MainActivity.class.getSimpleName();

    private int mStart = 1;
    private HandlerThread mTimerThread;
    private Handler mTimerHandler;
    private Runnable mRunnable = new Runnable() {
        @Override
        public void run() {
            mTimerHandler.postDelayed(mRunnable, 1000);

            MainActivity.this.runOnUiThread(
                    () -> {
                        mButton.setText(String.format("Listening - %ds left", AUDIO_LEN_IN_SECOND - mStart));
                        mStart += 1;
                    });
        }
    };

    @Override
    protected void onDestroy() {
        stopTimerThread();
        super.onDestroy();
    }

    protected void stopTimerThread() {
        mTimerThread.quitSafely();
        try {
            mTimerThread.join();
            mTimerThread = null;
            mTimerHandler = null;
            mStart = 1;
        } catch (InterruptedException e) {
            Log.e(TAG, "Error on stopping background thread", e);
        }
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mButton = findViewById(R.id.btnRecognize);
        mTextView = findViewById(R.id.tvResult);

        mButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButton.setText(String.format("Listening - %ds left", AUDIO_LEN_IN_SECOND));
                mButton.setEnabled(false);

                Thread thread = new Thread(MainActivity.this);
                thread.start();

                mTimerThread = new HandlerThread("Timer");
                mTimerThread.start();
                mTimerHandler = new Handler(mTimerThread.getLooper());
                mTimerHandler.postDelayed(mRunnable, 1000);

            }
        });
        requestMicrophonePermission();
    }

    private void requestMicrophonePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(
                    new String[]{android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
        }
    }

    private String assetFilePath(Context context, String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        } catch (IOException e) {
            Log.e(TAG, assetName + ": " + e.getLocalizedMessage());
        }
        return null;
    }

    private void showTranslationResult(String result) {
        mTextView.setText(result);
    }

    public void run() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

        int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        AudioRecord record = new AudioRecord(MediaRecorder.AudioSource.DEFAULT, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                bufferSize);

        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(LOG_TAG, "Audio Record can't initialize!");
            return;
        }
        record.startRecording();

        long shortsRead = 0;
        int recordingOffset = 0;
        short[] audioBuffer = new short[bufferSize / 2];
        short[] recordingBuffer = new short[RECORDING_LENGTH];

        while (shortsRead < RECORDING_LENGTH) {
            int numberOfShort = record.read(audioBuffer, 0, audioBuffer.length);
            shortsRead += numberOfShort;
            System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, numberOfShort);
            recordingOffset += numberOfShort;
        }

        record.stop();
        record.release();
        stopTimerThread();

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mButton.setText("Recognizing...");
            }
        });

        float[] floatInputBuffer = new float[RECORDING_LENGTH];

        // feed in float values between -1.0f and 1.0f by dividing the signed 16-bit inputs.
        for (int i = 0; i < RECORDING_LENGTH; ++i) {
            floatInputBuffer[i] = recordingBuffer[i] / (float)Short.MAX_VALUE;
        }

        final String result = recognize(floatInputBuffer);

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                showTranslationResult(result);
                mButton.setEnabled(true);
                mButton.setText("Start");
            }
        });
    }





    private String recognize(float[] floatInputBuffer) {
        if (module == null) {
            module = LiteModuleLoader.load(assetFilePath(getApplicationContext(), "ts_model2.ptl"));
        }

        // Play the audio data from buffer
        int sampleRate = 8000;
        int bufferSize = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT);
        AudioTrack audioTrack = new AudioTrack(AudioManager.STREAM_MUSIC, sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize, AudioTrack.MODE_STREAM);
        audioTrack.play();
        short[] shortInputBuffer = new short[floatInputBuffer.length];
        for (int i = 0; i < floatInputBuffer.length; i++) {
            shortInputBuffer[i] = (short) (floatInputBuffer[i] * Short.MAX_VALUE);
        }
        audioTrack.write(shortInputBuffer, 0, shortInputBuffer.length);
        audioTrack.stop();
        audioTrack.release();

        // Pass the audio data to the model for prediction
        double wav2vecinput[] = new double[RECORDING_LENGTH];
        for (int n = 0; n < RECORDING_LENGTH; n++)
            wav2vecinput[n] = floatInputBuffer[n];

        FloatBuffer inTensorBuffer = Tensor.allocateFloatBuffer(RECORDING_LENGTH);
        for (double val : wav2vecinput)
            inTensorBuffer.put((float)val);

        Tensor inTensor = Tensor.fromBlob(inTensorBuffer, new long[]{1, 1, RECORDING_LENGTH});
        Tensor outTensor = module.forward(IValue.from(inTensor)).toTensor();
        float[] outArray = outTensor.getDataAsFloatArray();

        // Define the list of labels
        String[] labels = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"};

        // Get the index of the maximum value in the output array
        int maxIndex = 0;
        for (int i = 1; i < outArray.length; i++) {
            if (outArray[i] > outArray[maxIndex]) {
                maxIndex = i;
            }
        }

        // Look up the corresponding label from the list
        String result = labels[maxIndex];

        return result;
    }

}


//
//
//
//    private String recognize(float[] floatInputBuffer) {
//        if (module == null) {
//            module = LiteModuleLoader.load(assetFilePath(getApplicationContext(), "ts_model2.ptl"));
//        }
//
//        // Save the audio data to a file
//        String fileName = "recording.wav";
//        File outputFile = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), fileName);
//        try {
//            FileOutputStream fos = new FileOutputStream(outputFile);
//            byte[] buffer = new byte[floatInputBuffer.length * 2];
//            for (int i = 0; i < floatInputBuffer.length; i++) {
//                short val = (short) (floatInputBuffer[i] * Short.MAX_VALUE);
//                buffer[i * 2] = (byte) (val & 0xff);
//                buffer[i * 2 + 1] = (byte) ((val >> 8) & 0xff);
//            }
//            fos.write(new byte[]{'R', 'I', 'F', 'F'});
//            fos.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(36 + buffer.length).array());
//            fos.write(new byte[]{'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
//            fos.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(16).array());
//            fos.write(ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort((short) 1).array());
//            fos.write(ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort((short) 1).array());
//            fos.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(16000).array());
//            fos.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(32000).array());
//            fos.write(ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort((short) 2).array());
//            fos.write(ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort((short) 16).array());
//            fos.write(new byte[]{'d', 'a', 't', 'a'});
//            fos.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(buffer.length).array());
//            fos.write(buffer);
//            fos.close();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//        // Pass the audio data to the model for prediction
//        double wav2vecinput[] = new double[RECORDING_LENGTH];
//        for (int n = 0; n < RECORDING_LENGTH; n++)
//            wav2vecinput[n] = floatInputBuffer[n];
//
//        FloatBuffer inTensorBuffer = Tensor.allocateFloatBuffer(RECORDING_LENGTH);
//        for (double val : wav2vecinput)
//            inTensorBuffer.put((float) val);
//
//        Tensor inTensor = Tensor.fromBlob(inTensorBuffer, new long[]{1, 1, RECORDING_LENGTH});
//        Tensor outTensor = module.forward(IValue.from(inTensor)).toTensor();
//        float[] outArray = outTensor.getDataAsFloatArray();
//
//        // Define the list of labels
//        String[] labels = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"};
//
//        // Get the index of the maximum value in the output array
//        int maxIndex = 0;
//        for (int i = 1; i < outArray.length; i++) {
//            if (outArray[i] > outArray[maxIndex]) {
//                maxIndex = i;
//            }
//        }
//
//        // Look up the corresponding label from the list
//        String result = labels[maxIndex];
//
//        return result;
//    }



//
//    private String recognize(float[] floatInputBuffer) {
//        if (module == null) {
//            module = LiteModuleLoader.load(assetFilePath(getApplicationContext(), "ts_model2.ptl"));
//        }
//
//        // Play the audio data from buffer
//        int sampleRate = 16000;
//        int bufferSize = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT);
//        AudioTrack audioTrack = new AudioTrack(AudioManager.STREAM_MUSIC, sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize, AudioTrack.MODE_STREAM);
//        audioTrack.play();
//        short[] shortInputBuffer = new short[floatInputBuffer.length];
//        for (int i = 0; i < floatInputBuffer.length; i++) {
//            shortInputBuffer[i] = (short) (floatInputBuffer[i] * Short.MAX_VALUE);
//        }
//        audioTrack.write(shortInputBuffer, 0, shortInputBuffer.length);
//        audioTrack.stop();
//        audioTrack.release();
//
//        // Pass the audio data to the model for prediction
//        double wav2vecinput[] = new double[RECORDING_LENGTH];
//        for (int n = 0; n < RECORDING_LENGTH; n++)
//            wav2vecinput[n] = floatInputBuffer[n];
//
//        FloatBuffer inTensorBuffer = Tensor.allocateFloatBuffer(RECORDING_LENGTH);
//        for (double val : wav2vecinput)
//            inTensorBuffer.put((float)val);
//
//        Tensor inTensor = Tensor.fromBlob(inTensorBuffer, new long[]{1, 1, RECORDING_LENGTH});
//        Tensor outTensor = module.forward(IValue.from(inTensor)).toTensor();
//        float[] outArray = outTensor.getDataAsFloatArray();
//
//        // Define the list of labels
//        String[] labels = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"};
//
//        // Get the index of the maximum value in the output array
//        int maxIndex = 0;
//        for (int i = 1; i < outArray.length; i++) {
//            if (outArray[i] > outArray[maxIndex]) {
//                maxIndex = i;
//            }
//        }
//
//        // Look up the corresponding label from the list
//        String result = labels[maxIndex];
//
//        return result;
//    }