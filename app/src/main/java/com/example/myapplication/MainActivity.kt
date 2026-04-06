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
    private var ttsReady = false

    data class Detection(
        val box: RectF,
        val labelId: Int,
        val labelName: String,
        val score: Float
    )

    private val labelTranslation = mapOf(
        "person" to "行人",
        "bicycle" to "自行车",
        "car" to "汽车",
        "motorcycle" to "摩托车",
        "bus" to "公共汽车",
        "truck" to "卡车",
        "traffic light" to "红绿灯",
        "stop sign" to "停止标志",
        "bench" to "长椅",
        "chair" to "椅子",
        "couch" to "沙发",
        "dining table" to "桌子",
        "bottle" to "瓶子",
        "cup" to "杯子",
        "door" to "门",
        "stairs" to "楼梯",
        "cat" to "猫",
        "dog" to "狗",
        "laptop" to "笔记本电脑",
        "mouse" to "鼠标",
        "keyboard" to "键盘"
    )

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) setupComposeUi()
        else Toast.makeText(this, "需要相机权限", Toast.LENGTH_SHORT).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val modelBytes = assets.open("yolov10n.onnx").readBytes()
            ortSession = ortEnv.createSession(modelBytes, OrtSession.SessionOptions())

            Log.d("VisionPath", "✅ 模型加载成功")
            Log.d("VisionPath", "模型输入: ${ortSession.inputNames}")
            Log.d("VisionPath", "模型输出: ${ortSession.outputNames}")

            tts = TextToSpeech(this, this)
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        } catch (e: Exception) {
            Log.e("VisionPath", "❌ 初始化失败: ${e.message}", e)
            Toast.makeText(this, "模型加载失败: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts.setLanguage(Locale.SIMPLIFIED_CHINESE)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("VisionPath", "⚠️ 中文TTS不支持，使用默认语言")
                tts.setLanguage(Locale.getDefault())
            }
            ttsReady = true
            Log.d("VisionPath", "✅ TTS初始化成功")
            tts.speak("导盲系统已启动，请问您要去哪里？", TextToSpeech.QUEUE_FLUSH, null, "init")
        } else {
            Log.e("VisionPath", "❌ TTS初始化失败，status=$status")
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

        val labels = remember {
            listOf(
                "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
                "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
                "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
                "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
                "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
                "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
                "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
                "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
            )
        }

        val previewView = remember {
            PreviewView(context).apply {
                scaleType = PreviewView.ScaleType.FILL_CENTER
            }
        }

        val cameraInitialized = remember { mutableStateOf(false) }

        LaunchedEffect(Unit) {
            if (cameraInitialized.value) return@LaunchedEffect

            val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

            cameraProviderFuture.addListener({
                if (cameraInitialized.value) return@addListener

                val cameraProvider = cameraProviderFuture.get()

                val preview = androidx.camera.core.Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                // ✅ 关键修复1：强制指定输出格式为 RGBA_8888，避免 YUV_420_888 导致 toBitmap() 返回 null
                val imageAnalysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(640, 640))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888) // ✅ 新增
                    .build()

                imageAnalysis.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                    Log.d("VisionPath", "📷 收到帧: ${imageProxy.width}x${imageProxy.height}, format=${imageProxy.format}")
                    processImageForOnnx(imageProxy, ortEnv, ortSession, labels) { results ->
                        detectionState.value = results
                        onDetectionResult(results)
                    }
                }

                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        imageAnalysis
                    )
                    cameraInitialized.value = true
                    Log.d("VisionPath", "✅ 摄像头绑定成功")
                } catch (e: Exception) {
                    Log.e("VisionPath", "❌ 相机绑定失败: ${e.message}", e)
                }

            }, ContextCompat.getMainExecutor(context))
        }

        Box(modifier = Modifier.fillMaxSize()) {
            AndroidView(
                factory = { previewView },
                modifier = Modifier.fillMaxSize()
            )
            DetectionOverlay(detectionState.value)
        }
    }

    // ✅ 关键修复2：将 ImageProxy 转为 Bitmap 的正确方式（兼容 RGBA_8888 和 YUV 两种格式）
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            // 优先路径：已是 RGBA_8888，可以直接用 planes[0]
            val plane = imageProxy.planes[0]
            val buffer = plane.buffer
            val pixelStride = plane.pixelStride
            val rowStride = plane.rowStride
            val rowPadding = rowStride - pixelStride * imageProxy.width

            Log.d("VisionPath", "帧信息: format=${imageProxy.format}, pixelStride=$pixelStride, rowStride=$rowStride, rowPadding=$rowPadding")

            val bitmap = Bitmap.createBitmap(
                imageProxy.width + rowPadding / pixelStride,
                imageProxy.height,
                Bitmap.Config.ARGB_8888
            )
            bitmap.copyPixelsFromBuffer(buffer)

            // 裁剪掉 rowPadding 多出来的宽度
            val cropped = if (rowPadding > 0) {
                Bitmap.createBitmap(bitmap, 0, 0, imageProxy.width, imageProxy.height)
            } else {
                bitmap
            }

            // 按相机旋转角度矫正方向
            val rotation = imageProxy.imageInfo.rotationDegrees
            if (rotation != 0) {
                val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                val rotated = Bitmap.createBitmap(cropped, 0, 0, cropped.width, cropped.height, matrix, true)
                if (rotated !== cropped) cropped.recycle()
                rotated
            } else {
                cropped
            }
        } catch (e: Exception) {
            Log.e("VisionPath", "❌ Bitmap转换失败: ${e.message}", e)
            null
        }
    }

    private fun processImageForOnnx(
        imageProxy: ImageProxy,
        ortEnv: OrtEnvironment,
        ortSession: OrtSession,
        labels: List<String>,
        onResults: (List<Detection>) -> Unit
    ) {
        try {
            Log.d("VisionPath", "🔄 开始处理帧...")

            // ✅ 使用新的转换方法，不再依赖可能返回 null 的 toBitmap()
            val rawBitmap = imageProxyToBitmap(imageProxy)
            if (rawBitmap == null) {
                Log.e("VisionPath", "❌ Bitmap获取失败，跳过本帧")
                return
            }

            val bitmap = Bitmap.createScaledBitmap(rawBitmap, 640, 640, true)
            if (rawBitmap !== bitmap) rawBitmap.recycle()
            Log.d("VisionPath", "✅ Bitmap缩放完成: ${bitmap.width}x${bitmap.height}")

            val numPixels = 640 * 640
            val floatBuffer = FloatArray(3 * numPixels)
            val pixels = IntArray(numPixels)
            bitmap.getPixels(pixels, 0, 640, 0, 0, 640, 640)
            bitmap.recycle()

            // CHW 格式：先 R 通道，再 G，再 B
            for (i in 0 until numPixels) {
                val color = pixels[i]
                floatBuffer[i]                 = (color shr 16 and 0xFF) / 255.0f  // R
                floatBuffer[numPixels + i]     = (color shr 8  and 0xFF) / 255.0f  // G
                floatBuffer[2 * numPixels + i] = (color        and 0xFF) / 255.0f  // B
            }

            Log.d("VisionPath", "✅ 像素转换完成，开始推理...")

            val inputName = ortSession.inputNames.iterator().next()
            val inputTensor = OnnxTensor.createTensor(
                ortEnv,
                java.nio.FloatBuffer.wrap(floatBuffer),
                longArrayOf(1, 3, 640, 640)
            )

            inputTensor.use {
                val results = ortSession.run(Collections.singletonMap(inputName, inputTensor))
                results.use { output ->
                    val outputTensor = output[0] as OnnxTensor
                    val shape = outputTensor.info.shape
                    val buffer = outputTensor.floatBuffer
                    val outputArray = FloatArray(buffer.remaining()).also { buffer.get(it) }

                    Log.d("VisionPath", "✅ 推理完成，输出shape: ${shape.toList()}")
                    Log.d("VisionPath", "前10个值: ${outputArray.take(10)}")

                    val detections = mutableListOf<Detection>()

                    if (shape.size == 3) {
                        val dim1 = shape[1].toInt()
                        val dim2 = shape[2].toInt()

                        Log.d("VisionPath", "dim1=$dim1, dim2=$dim2")

                        if (dim2 == 6) {
                            // 格式 [1, 300, 6]：每行是 [x1, y1, x2, y2, score, classId]
                            Log.d("VisionPath", "解析格式: [1, N, 6]")
                            for (i in 0 until dim1) {
                                val offset = i * 6
                                val score = outputArray[offset + 4]
                                if (score > 0.25f) {  // ✅ 提高阈值到0.25，减少误报
                                    val labelId = outputArray[offset + 5].toInt().coerceIn(0, labels.size - 1)
                                    detections.add(Detection(
                                        RectF(
                                            outputArray[offset],
                                            outputArray[offset + 1],
                                            outputArray[offset + 2],
                                            outputArray[offset + 3]
                                        ),
                                        labelId,
                                        labels[labelId],
                                        score
                                    ))
                                }
                            }
                        } else if (dim1 == 6) {
                            // 格式 [1, 6, 300]
                            Log.d("VisionPath", "解析格式: [1, 6, N]")
                            for (i in 0 until dim2) {
                                val score = outputArray[4 * dim2 + i]
                                if (score > 0.25f) {
                                    val labelId = outputArray[5 * dim2 + i].toInt().coerceIn(0, labels.size - 1)
                                    detections.add(Detection(
                                        RectF(
                                            outputArray[i],
                                            outputArray[dim2 + i],
                                            outputArray[2 * dim2 + i],
                                            outputArray[3 * dim2 + i]
                                        ),
                                        labelId,
                                        labels[labelId],
                                        score
                                    ))
                                }
                            }
                        } else {
                            Log.w("VisionPath", "⚠️ 未知输出格式: dim1=$dim1, dim2=$dim2，请确认模型输出")
                        }
                    } else if (shape.size == 2) {
                        // 某些导出版本可能是 [300, 6]（没有 batch 维度）
                        val rows = shape[0].toInt()
                        val cols = shape[1].toInt()
                        Log.d("VisionPath", "解析格式: [N, 6] rows=$rows cols=$cols")
                        if (cols == 6) {
                            for (i in 0 until rows) {
                                val offset = i * 6
                                val score = outputArray[offset + 4]
                                if (score > 0.25f) {
                                    val labelId = outputArray[offset + 5].toInt().coerceIn(0, labels.size - 1)
                                    detections.add(Detection(
                                        RectF(
                                            outputArray[offset],
                                            outputArray[offset + 1],
                                            outputArray[offset + 2],
                                            outputArray[offset + 3]
                                        ),
                                        labelId,
                                        labels[labelId],
                                        score
                                    ))
                                }
                            }
                        }
                    } else {
                        Log.w("VisionPath", "⚠️ 无法识别的shape维度: ${shape.toList()}")
                    }

                    Log.d("VisionPath", "✅ 有效检测数量: ${detections.size}")
                    if (detections.isNotEmpty()) {
                        Log.d("VisionPath", "检测到: ${detections.map { "${it.labelName}(${String.format("%.2f", it.score)})" }}")
                    }

                    runOnUiThread { onResults(detections) }
                }
            }
        } catch (e: Exception) {
            Log.e("VisionPath", "❌ 推理错误: ${e.message}", e)
        } finally {
            imageProxy.close()
        }
    }

    private fun handleDetections(detections: List<Detection>) {
        if (detections.isEmpty()) return
        if (!ttsReady) return

        val dangerPriority = listOf("car", "bus", "truck", "motorcycle", "bicycle", "person", "chair", "bottle", "laptop")

        val dangerObject = detections
            .filter { it.labelName in dangerPriority }
            .maxByOrNull { dangerPriority.indexOf(it.labelName) * -1 + it.score }

        if (dangerObject != null) {
            val currentTime = System.currentTimeMillis()
            if (currentTime - lastSpeakTime > 3000) {
                val chineseName = labelTranslation[dangerObject.labelName] ?: dangerObject.labelName
                val position = getPositionHint(dangerObject.box)
                val message = "注意，${position}有${chineseName}"
                Log.d("VisionPath", "🔊 播报: $message")
                tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, "detection_${currentTime}")
                lastSpeakTime = currentTime
            }
        }
    }

    private fun getPositionHint(box: RectF): String {
        val centerX = (box.left + box.right) / 2
        return when {
            centerX < 640 * 0.33f -> "左前方"
            centerX > 640 * 0.66f -> "右前方"
            else -> "正前方"
        }
    }

    @Composable
    fun DetectionOverlay(detections: List<Detection>) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            detections.forEach { detection ->
                val box = detection.box
                val scaleX = size.width / 640f
                val scaleY = size.height / 640f

                val left   = box.left   * scaleX
                val top    = box.top    * scaleY
                val right  = box.right  * scaleX
                val bottom = box.bottom * scaleY

                val boxPaint = Paint().apply {
                    color = Color.RED
                    style = Paint.Style.STROKE
                    strokeWidth = 6f
                }
                drawContext.canvas.nativeCanvas.drawRect(left, top, right, bottom, boxPaint)

                val textPaint = Paint().apply {
                    color = Color.RED
                    textSize = 36f
                    isFakeBoldText = true
                    setShadowLayer(4f, 2f, 2f, Color.BLACK)
                }
                val chineseName = labelTranslation[detection.labelName] ?: detection.labelName
                val label = "$chineseName ${"%.0f".format(detection.score * 100)}%"
                drawContext.canvas.nativeCanvas.drawText(label, left, top - 8f, textPaint)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::ortSession.isInitialized) ortSession.close()
        if (::ortEnv.isInitialized) ortEnv.close()
        if (::tts.isInitialized) { tts.stop(); tts.shutdown() }
    }
}
