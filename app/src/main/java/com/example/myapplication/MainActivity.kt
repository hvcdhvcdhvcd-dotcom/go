package com.example.myapplication

import android.Manifest
import android.graphics.*
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.util.Size
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.example.myapplication.ui.theme.MyApplicationTheme
import java.util.*
import java.util.concurrent.Executors

class MainActivity : ComponentActivity(), TextToSpeech.OnInitListener {

    private lateinit var ortEnv: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private lateinit var tts: TextToSpeech
    private val detectionState = mutableStateOf<List<Detection>>(emptyList())
    private var lastSpeakTime = 0L

    data class Detection(
        val box: RectF,
        val labelId: Int,
        val labelName: String,
        val score: Float
    )

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { if (it) setupComposeUi() else Toast.makeText(this, "需相机权限", Toast.LENGTH_SHORT).show() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val modelBytes = assets.open("yolov10n.onnx").readBytes()
            ortSession = ortEnv.createSession(modelBytes, OrtSession.SessionOptions())
            tts = TextToSpeech(this, this)
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        } catch (e: Exception) {
            Log.e("VisionPath", "初始化失败: ${e.message}")
            Toast.makeText(this, "模型加载失败", Toast.LENGTH_LONG).show()
        }
    }

    private fun setupComposeUi() {
        setContent {
            MyApplicationTheme {
                BlindAssistantScreen(ortEnv, ortSession, detectionState) { results ->
                    handleDetections(results)
                }
            }
        }
    }

    @Composable
    fun BlindAssistantScreen(
        ortEnv: OrtEnvironment,
        ortSession: OrtSession,
        detectionState: MutableState<List<Detection>>,
        onDetectionResult: (List<Detection>) -> Unit
    ) {
        val context = LocalContext.current
        val lifecycleOwner = LocalLifecycleOwner.current
        val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }

        val labels = remember { listOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
            "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        )}

        Box(modifier = Modifier.fillMaxSize()) {
            AndroidView(
                factory = { ctx ->
                    PreviewView(ctx).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
                },
                modifier = Modifier.fillMaxSize()
            ) { previewView ->
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = androidx.camera.core.Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                    // 核心修改：只定义一个 imageAnalysis 实例
                    val imageAnalysis = ImageAnalysis.Builder()
                        .setTargetResolution(Size(640, 640))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()

                    imageAnalysis.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                        processImageForOnnx(imageProxy, ortEnv, ortSession, labels) { results ->
                            detectionState.value = results
                            onDetectionResult(results)
                        }
                    }

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis)
                    } catch (e: Exception) { Log.e("VisionPath", "绑定失败", e) }
                }, ContextCompat.getMainExecutor(context))
            }

            DetectionOverlay(detectionState.value)
        }
    }

    private fun processImageForOnnx(
        imageProxy: ImageProxy,
        ortEnv: OrtEnvironment,
        ortSession: OrtSession,
        labels: List<String>,
        onResults: (List<Detection>) -> Unit
    ) {
        val bitmap = imageProxy.toBitmap()?.let { Bitmap.createScaledBitmap(it, 640, 640, true) } ?: return
        val numPixels = 640 * 640
        val floatBuffer = FloatArray(3 * numPixels)
        val pixels = IntArray(numPixels)
        bitmap.getPixels(pixels, 0, 640, 0, 0, 640, 640)

        for (i in 0 until numPixels) {
            val color = pixels[i]
            floatBuffer[i] = (color shr 16 and 0xFF) / 255.0f
            floatBuffer[numPixels + i] = (color shr 8 and 0xFF) / 255.0f
            floatBuffer[2 * numPixels + i] = (color and 0xFF) / 255.0f
        }

        try {
            val inputName = ortSession.inputNames.iterator().next()
            val inputTensor = OnnxTensor.createTensor(ortEnv, java.nio.FloatBuffer.wrap(floatBuffer), longArrayOf(1, 3, 640, 640))

            inputTensor.use {
                val results = ortSession.run(Collections.singletonMap(inputName, inputTensor))
                results.use { output ->
                    if (output.size() == 0) return@use
                    val outputTensor = output[0] as OnnxTensor
                    val shape = outputTensor.info.shape // [1, 300, 6] 或 [1, 6, 300]
                    val buffer = outputTensor.floatBuffer
                    val outputArray = FloatArray(buffer.remaining())
                    buffer.get(outputArray)

                    val detections = mutableListOf<Detection>()

                    // 兼容性逻辑：自动判断输出矩阵的维度
                    if (shape.size == 3) {
                        val dim1 = shape[1].toInt() // 可能是 300
                        val dim2 = shape[2].toInt() // 可能是 6

                        if (dim2 == 6) { // 格式 [1, 300, 6]
                            for (i in 0 until dim1) {
                                val offset = i * 6
                                val score = outputArray[offset + 4]
                                if (score > 0.25f) { // 调低阈值方便测试
                                    val labelId = outputArray[offset + 5].toInt()
                                    detections.add(Detection(
                                        RectF(outputArray[offset], outputArray[offset+1], outputArray[offset+2], outputArray[offset+3]),
                                        labelId, labels.getOrElse(labelId){"?"}, score
                                    ))
                                }
                            }
                        } else if (dim1 == 6) { // 格式 [1, 6, 300]
                            for (i in 0 until dim2) {
                                val score = outputArray[4 * dim2 + i]
                                if (score > 0.25f) {
                                    val labelId = outputArray[5 * dim2 + i].toInt()
                                    detections.add(Detection(
                                        RectF(outputArray[0*dim2+i], outputArray[1*dim2+i], outputArray[2*dim2+i], outputArray[3*dim2+i]),
                                        labelId, labels.getOrElse(labelId){"?"}, score
                                    ))
                                }
                            }
                        }
                    }
                    runOnUiThread { onResults(detections) }
                }
            }
        } catch (e: Exception) { Log.e("VisionPath", "推理错误: ${e.message}") }
        finally { imageProxy.close() }
    }

    private fun handleDetections(detections: List<Detection>) {
        if (detections.isEmpty()) return

        // 增加调试 Log，确保逻辑在走
        Log.d("VisionPath", "当前检测到物体数量: ${detections.size}")

        val dangerObject = detections
            .filter { it.labelName in listOf("person", "car", "chair", "bottle", "bicycle", "bus") }
            .maxByOrNull { it.score }

        if (dangerObject != null) {
            val currentTime = System.currentTimeMillis()
            if (currentTime - lastSpeakTime > 4000) {
                tts.speak("前方发现${dangerObject.labelName}", TextToSpeech.QUEUE_FLUSH, null)
                lastSpeakTime = currentTime
            }
        }
    }

    @Composable
    fun DetectionOverlay(detections: List<Detection>) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val paint = Paint().apply {
                color = Color.RED
                style = Paint.Style.STROKE
                strokeWidth = 8f
            }
            detections.forEach {
                // 自动判断坐标是否需要缩放 (YOLO 坐标可能是 0~1，也可能是 0~640)
                val rect = if (it.box.left <= 1.1f) {
                    RectF(it.box.left * size.width, it.box.top * size.height, it.box.right * size.width, it.box.bottom * size.height)
                } else {
                    RectF(it.box.left / 640 * size.width, it.box.top / 640 * size.height, it.box.right / 640 * size.width, it.box.bottom / 640 * size.height)
                }
                drawContext.canvas.nativeCanvas.drawRect(rect, paint)
            }
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.CHINESE
            tts.speak("系统启动", TextToSpeech.QUEUE_FLUSH, null)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::ortSession.isInitialized) ortSession.close()
        if (::ortEnv.isInitialized) ortEnv.close()
        if (::tts.isInitialized) tts.shutdown()
    }
}