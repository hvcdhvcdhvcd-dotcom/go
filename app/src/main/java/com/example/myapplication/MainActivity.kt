package com.example.myapplication

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
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
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size as ComposeSize
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.example.myapplication.ui.theme.MyApplicationTheme
import java.util.*
import java.util.concurrent.Executors

// ════════════════════════════════════════════════════════════
// 应用模式
// ════════════════════════════════════════════════════════════
enum class AppMode {
    IDLE,        // 等待语音指令
    NAVIGATION,  // 高德地图导航（预留）
    DETECTION,   // YOLOv10 实时检测
    AI_SCENE     // AI 场景识别（预留）
}

class MainActivity : ComponentActivity(), TextToSpeech.OnInitListener {

    // ── ONNX ──────────────────────────────────────────────
    private lateinit var ortEnv: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private var modelReady = false

    // ── TTS ───────────────────────────────────────────────
    private lateinit var tts: TextToSpeech
    private var ttsReady = false

    // ── UI 状态 ────────────────────────────────────────────
    // cameraReadyState 控制相机何时绑定（权限通过后翻转为 true）
    // resumeToken 每次 onResume 自增，强制 LaunchedEffect 重新绑定相机
    private val detectionState   = mutableStateOf<List<Detection>>(emptyList())
    private val appModeState     = mutableStateOf(AppMode.DETECTION) // 默认检测模式
    private val statusTextState  = mutableStateOf("正在初始化…")
    private val cameraReadyState = mutableStateOf(false)             // 相机绑定开关
    private val resumeToken      = mutableStateOf(0)                 // 每次 onResume +1，触发相机重绑

    // ── 其他 ──────────────────────────────────────────────
    private var lastSpeakTime          = 0L
    private var currentRotationDegrees = 0
    private var speechRecognizer: SpeechRecognizer? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    private var isRequestingVoiceCommand = false // 标记用户是否正在请求语音指令

    // ════════════════════════════════════════════════════════
    // 数据类
    // ════════════════════════════════════════════════════════
    data class Detection(
        val box: RectF,          // 归一化坐标 [0,1]，已变换到屏幕坐标系
        val labelId: Int,
        val labelName: String,
        val score: Float
    )

    // ════════════════════════════════════════════════════════
    // COCO 标签
    // ════════════════════════════════════════════════════════
    private val labels = listOf(
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
        "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
        "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
        "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
        "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
        "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
        "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
        "scissors","teddy bear","hair drier","toothbrush"
    )

    private val labelZh = mapOf(
        "person" to "行人","bicycle" to "自行车","car" to "汽车","motorcycle" to "摩托车",
        "bus" to "公共汽车","truck" to "卡车","traffic light" to "红绿灯","stop sign" to "停止标志",
        "bench" to "长椅","chair" to "椅子","couch" to "沙发","dining table" to "桌子",
        "bottle" to "瓶子","cup" to "杯子","cat" to "猫","dog" to "狗",
        "laptop" to "笔记本电脑","mouse" to "鼠标","keyboard" to "键盘","cell phone" to "手机",
        "tv" to "电视","bed" to "床","toilet" to "马桶","sink" to "水槽","refrigerator" to "冰箱",
        "backpack" to "背包","umbrella" to "雨伞","book" to "书","clock" to "时钟",
        "potted plant" to "盆栽","scissors" to "剪刀","fire hydrant" to "消防栓",
        "bird" to "鸟","horse" to "马","cow" to "牛","dog" to "狗"
    )

    private val dangerPriority = listOf(
        "car","bus","truck","motorcycle","bicycle","person",
        "traffic light","stop sign","chair","couch","dining table","bottle","laptop"
    )

    // ════════════════════════════════════════════════════════
    // 权限请求
    // ════════════════════════════════════════════════════════
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { perms ->
        val cameraOk = perms[Manifest.permission.CAMERA] == true
        val audioOk  = perms[Manifest.permission.RECORD_AUDIO] == true
        val locationOk = perms[Manifest.permission.ACCESS_FINE_LOCATION] == true || perms[Manifest.permission.ACCESS_COARSE_LOCATION] == true
        Log.d(TAG, "权限结果 → 相机=$cameraOk  麦克风=$audioOk  定位=$locationOk")
        if (cameraOk) {
            cameraReadyState.value = true   // 触发 Compose 重组，启动相机
        } else {
            Toast.makeText(this, "需要相机权限才能检测障碍物", Toast.LENGTH_LONG).show()
            statusTextState.value = "❌ 缺少相机权限"
        }
        if (!audioOk) {
            statusTextState.value = "障碍物检测运行中（语音功能需麦克风权限）"
        } else {
            // 如果用户刚刚授予了麦克风权限，并且当前正在请求语音指令，则重新启动语音识别
            if (isRequestingVoiceCommand) {
                isRequestingVoiceCommand = false
                mainHandler.postDelayed({ startVoiceCommand() }, 500)
            }
        }
        if (!locationOk) {
            Toast.makeText(this, "导航功能需要定位权限", Toast.LENGTH_LONG).show()
        }
    }

    // ════════════════════════════════════════════════════════
    // onCreate
    // ════════════════════════════════════════════════════════
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // 1. 加载 ONNX 模型
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val bytes = assets.open("yolov10n.onnx").readBytes()
            ortSession = ortEnv.createSession(bytes, OrtSession.SessionOptions())
            modelReady = true
            Log.d(TAG, "✅ 模型加载成功  输入=${ortSession.inputNames}  输出=${ortSession.outputNames}")
        } catch (e: Exception) {
            Log.e(TAG, "❌ 模型加载失败: ${e.message}", e)
            Toast.makeText(this, "模型加载失败: ${e.message}", Toast.LENGTH_LONG).show()
        }

        // 2. 初始化 TTS
        tts = TextToSpeech(this, this)

        // ✅ 关键修复：先渲染 UI，解决白屏问题
        setContent {
            MyApplicationTheme {
                MainScreen()
            }
        }

        // 3. 检查权限，已授予则直接开启相机，否则弹请求
        val need = mutableListOf<String>()
        if (!hasPerm(Manifest.permission.CAMERA))       need += Manifest.permission.CAMERA
        if (!hasPerm(Manifest.permission.RECORD_AUDIO)) need += Manifest.permission.RECORD_AUDIO
        if (!hasPerm(Manifest.permission.ACCESS_FINE_LOCATION)) need += Manifest.permission.ACCESS_FINE_LOCATION
        if (!hasPerm(Manifest.permission.ACCESS_COARSE_LOCATION)) need += Manifest.permission.ACCESS_COARSE_LOCATION

        if (need.isEmpty()) {
            Log.d(TAG, "✅ 所有权限已授予，直接启动相机")
            cameraReadyState.value = true
        } else {
            permissionLauncher.launch(need.toTypedArray())
        }
    }

    private fun hasPerm(perm: String) =
        ContextCompat.checkSelfPermission(this, perm) == PackageManager.PERMISSION_GRANTED

    // ════════════════════════════════════════════════════════
    // TTS 回调
    // ════════════════════════════════════════════════════════
    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val r = tts.setLanguage(Locale.SIMPLIFIED_CHINESE)
            if (r == TextToSpeech.LANG_MISSING_DATA || r == TextToSpeech.LANG_NOT_SUPPORTED) {
                tts.setLanguage(Locale.getDefault())
                Log.w(TAG, "⚠️ 不支持中文TTS，已切换默认语言")
            }
            ttsReady = true
            Log.d(TAG, "✅ TTS 初始化成功")
            speak("导盲系统已启动，障碍物检测进行中")
            statusTextState.value = "障碍物检测运行中"
        } else {
            Log.e(TAG, "❌ TTS 初始化失败 status=$status")
        }
    }

    // ✅ 修复问题三：从导航返回时相机黑屏
    // onResume 时自增 resumeToken，触发 Compose LaunchedEffect 重新执行相机绑定
    override fun onResume() {
        super.onResume()
        if (cameraReadyState.value) {
            resumeToken.value += 1
            Log.d(TAG, "📷 onResume 触发相机重绑 token=${resumeToken.value}")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        speechRecognizer?.destroy()
        if (::ortSession.isInitialized) ortSession.close()
        if (::ortEnv.isInitialized)     ortEnv.close()
        if (::tts.isInitialized)        { tts.stop(); tts.shutdown() }
    }

    // ════════════════════════════════════════════════════════
    // Compose UI
    // ════════════════════════════════════════════════════════
    @Composable
    fun MainScreen() {
        val context        = LocalContext.current
        val lifecycleOwner = LocalLifecycleOwner.current
        val mode           by appModeState
        val statusText     by statusTextState
        val detections     by detectionState
        val cameraReady    by cameraReadyState   // 权限通过后变 true，触发相机绑定
        val resumeTok      by resumeToken        // 每次 onResume 自增，强制重绑

        val previewView = remember {
            PreviewView(context).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
        }

        // ✅ 修复问题三：key 同时监听 cameraReady 和 resumeTok
        // 返回 MainActivity 时 resumeTok 变化，相机重新绑定，解决黑屏
        LaunchedEffect(cameraReady, resumeTok) {
            if (!cameraReady) return@LaunchedEffect
            Log.d(TAG, "📷 开始绑定相机 resumeTok=$resumeTok")

            val future = ProcessCameraProvider.getInstance(context)
            future.addListener({
                runCatching {
                    val provider = future.get()
                    val preview  = androidx.camera.core.Preview.Builder().build()
                        .also { it.setSurfaceProvider(previewView.surfaceProvider) }

                    val analysis = ImageAnalysis.Builder()
                        .setTargetResolution(Size(640, 640))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()
                        .also { ia ->
                            ia.setAnalyzer(Executors.newSingleThreadExecutor()) { proxy ->
                                currentRotationDegrees = proxy.imageInfo.rotationDegrees
                                if (appModeState.value == AppMode.DETECTION && modelReady) {
                                    processFrame(proxy)
                                } else {
                                    detectionState.value = emptyList()
                                    proxy.close()
                                }
                            }
                        }

                    provider.unbindAll()
                    provider.bindToLifecycle(
                        lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
                    )
                    Log.d(TAG, "✅ 相机绑定成功")
                    mainHandler.post { statusTextState.value = "障碍物检测运行中" }

                }.onFailure { e ->
                    Log.e(TAG, "❌ 相机绑定失败: ${e.message}", e)
                    mainHandler.post { statusTextState.value = "相机启动失败: ${e.message}" }
                }
            }, ContextCompat.getMainExecutor(context))
        }

        // ── 布局 ─────────────────────────────────────────
        Box(modifier = Modifier.fillMaxSize()) {

            // 相机预览（始终铺满屏幕）
            AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())

            // 检测框（仅 DETECTION 模式）
            if (mode == AppMode.DETECTION) {
                DetectionOverlay(detections)
            }

            // 顶部模式标签
            Surface(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 52.dp),
                color = modeColor(mode).copy(alpha = 0.82f),
                shape = MaterialTheme.shapes.medium
            ) {
                Text(
                    modeLabel(mode),
                    color = Color.White,
                    fontSize = 14.sp,
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 5.dp)
                )
            }

            // 底部控制区
            Column(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                // 状态文字
                Surface(
                    color = androidx.compose.ui.graphics.Color(0xCC000000),
                    shape = MaterialTheme.shapes.medium
                ) {
                    Text(
                        statusText,
                        color = Color.White,
                        fontSize = 15.sp,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                    )
                }
                // 按钮
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Button(
                        onClick = { switchToDetection() },
                        colors = ButtonDefaults.buttonColors(
                            containerColor =
                                if (mode == AppMode.DETECTION) androidx.compose.ui.graphics.Color(0xFFD32F2F)
                                else androidx.compose.ui.graphics.Color(0xFF616161)
                        )
                    ) { Text("障碍检测") }

                    Button(
                        onClick = { startVoiceCommand() },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = androidx.compose.ui.graphics.Color(0xFF1565C0)
                        )
                    ) { Text("🎤 语音指令") }

                    Button(
                        onClick = {
                            val intent = Intent(context, IntegratedNaviActivity::class.java)
                            context.startActivity(intent)
                        },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = androidx.compose.ui.graphics.Color(0xFF388E3C)
                        )
                    ) { Text("实时检测障碍导航") }
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════
    // ✅ 检测框绘制（修复 Canvas 坐标系混用问题）
    //    边框用 Compose drawRect，文字用 nativeCanvas（支持中文）
    // ════════════════════════════════════════════════════════
    @Composable
    fun DetectionOverlay(detections: List<Detection>) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val cw = size.width
            val ch = size.height

            detections.forEach { det ->
                val b      = det.box
                val left   = (b.left   * cw).coerceIn(0f, cw)
                val top    = (b.top    * ch).coerceIn(0f, ch)
                val right  = (b.right  * cw).coerceIn(0f, cw)
                val bottom = (b.bottom * ch).coerceIn(0f, ch)
                val bw = right - left
                val bh = bottom - top
                if (bw < 1f || bh < 1f) return@forEach

                // 红色边框（纯 Compose，无坐标系冲突）
                drawRect(
                    color   = Color.Red,
                    topLeft = Offset(left, top),
                    size    = ComposeSize(bw, bh),
                    style   = Stroke(width = 5f)
                )

                // 标签背景 + 文字
                val zh    = labelZh[det.labelName] ?: det.labelName
                val label = "$zh ${"%.0f".format(det.score * 100)}%"
                val textPaint = android.graphics.Paint().apply {
                    color        = android.graphics.Color.WHITE
                    textSize     = 36f
                    isFakeBoldText = true
                    isAntiAlias  = true
                }
                val bgPaint = android.graphics.Paint().apply {
                    color = android.graphics.Color.argb(200, 180, 0, 0)
                }
                val bounds = android.graphics.Rect()
                textPaint.getTextBounds(label, 0, label.length, bounds)
                val labelTop = (top - bounds.height() - 6f).coerceAtLeast(0f)

                drawContext.canvas.nativeCanvas.apply {
                    drawRect(left, labelTop, left + bounds.width() + 10f, top, bgPaint)
                    drawText(label, left + 5f, top - 4f, textPaint)
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════
    // 帧推理
    // ════════════════════════════════════════════════════════
    private fun processFrame(proxy: ImageProxy) {
        try {
            val raw = imageProxyToBitmap(proxy) ?: return
            val lb  = letterboxBitmap(raw, 640)
            raw.recycle()

            val floats    = bitmapToFloatBuffer(lb)
            lb.recycle()

            val inputName = ortSession.inputNames.iterator().next()
            val tensor    = OnnxTensor.createTensor(
                ortEnv, java.nio.FloatBuffer.wrap(floats), longArrayOf(1, 3, 640, 640)
            )
            tensor.use {
                val out = ortSession.run(Collections.singletonMap(inputName, tensor))
                out.use { o ->
                    val t   = o[0] as OnnxTensor
                    val arr = FloatArray(t.floatBuffer.remaining()).also { t.floatBuffer.get(it) }
                    val raw2 = parseDetections(arr, t.info.shape)
                    val dets = raw2.map { d ->
                        Detection(rotateNormalizedBox(d.box, currentRotationDegrees),
                            d.labelId, d.labelName, d.score)
                    }
                    Log.d(TAG, "检测到 ${dets.size} 个物体  rotation=$currentRotationDegrees")
                    mainHandler.post {
                        detectionState.value = dets
                        handleDetections(dets)
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "❌ 推理错误: ${e.message}", e)
        } finally {
            proxy.close()
        }
    }

    // ════════════════════════════════════════════════════════
    // ✅ Letterbox 预处理（保持宽高比，提升准确率）
    // ════════════════════════════════════════════════════════
    private fun letterboxBitmap(src: Bitmap, target: Int): Bitmap {
        val scale  = minOf(target.toFloat() / src.width, target.toFloat() / src.height)
        val sw     = (src.width  * scale).toInt()
        val sh     = (src.height * scale).toInt()
        val padL   = (target - sw) / 2
        val padT   = (target - sh) / 2
        val dst    = Bitmap.createBitmap(target, target, Bitmap.Config.ARGB_8888)
        dst.eraseColor(android.graphics.Color.argb(255, 114, 114, 114))
        val c      = android.graphics.Canvas(dst)
        val scaled = Bitmap.createScaledBitmap(src, sw, sh, true)
        c.drawBitmap(scaled, padL.toFloat(), padT.toFloat(), null)
        if (scaled !== src) scaled.recycle()
        return dst
    }

    private fun bitmapToFloatBuffer(bmp: Bitmap): FloatArray {
        val n   = 640 * 640
        val buf = FloatArray(3 * n)
        val px  = IntArray(n)
        bmp.getPixels(px, 0, 640, 0, 0, 640, 640)
        for (i in 0 until n) {
            val c = px[i]
            buf[i]         = (c shr 16 and 0xFF) / 255f
            buf[n + i]     = (c shr 8  and 0xFF) / 255f
            buf[2 * n + i] = (c        and 0xFF) / 255f
        }
        return buf
    }

    private fun imageProxyToBitmap(proxy: ImageProxy): Bitmap? = runCatching {
        val plane  = proxy.planes[0]
        val ps     = plane.pixelStride
        val rs     = plane.rowStride
        val pad    = rs - ps * proxy.width
        val bmp    = Bitmap.createBitmap(proxy.width + pad / ps, proxy.height, Bitmap.Config.ARGB_8888)
        bmp.copyPixelsFromBuffer(plane.buffer)
        if (pad > 0) {
            val crop = Bitmap.createBitmap(bmp, 0, 0, proxy.width, proxy.height)
            bmp.recycle(); crop
        } else bmp
    }.getOrElse { e -> Log.e(TAG, "❌ Bitmap转换: ${e.message}"); null }

    // ════════════════════════════════════════════════════════
    // ONNX 输出解析
    // ════════════════════════════════════════════════════════
    private fun parseDetections(arr: FloatArray, shape: LongArray): List<Detection> {
        val list  = mutableListOf<Detection>()
        fun id(f: Float) = f.toInt().coerceIn(0, labels.size - 1)
        when {
            shape.size == 3 && shape[2].toInt() == 6 -> {   // [1, N, 6]
                repeat(shape[1].toInt()) { i ->
                    val o = i * 6; val s = arr[o + 4]
                    if (s > CONF_THRESHOLD) { val id = id(arr[o+5])
                        list += Detection(RectF(arr[o],arr[o+1],arr[o+2],arr[o+3]),id,labels[id],s) }
                }
            }
            shape.size == 3 && shape[1].toInt() == 6 -> {   // [1, 6, N]
                val n = shape[2].toInt()
                repeat(n) { i ->
                    val s = arr[4*n+i]
                    if (s > CONF_THRESHOLD) { val id = id(arr[5*n+i])
                        list += Detection(RectF(arr[i],arr[n+i],arr[2*n+i],arr[3*n+i]),id,labels[id],s) }
                }
            }
            shape.size == 2 && shape[1].toInt() == 6 -> {   // [N, 6]
                repeat(shape[0].toInt()) { i ->
                    val o = i * 6; val s = arr[o + 4]
                    if (s > CONF_THRESHOLD) { val id = id(arr[o+5])
                        list += Detection(RectF(arr[o],arr[o+1],arr[o+2],arr[o+3]),id,labels[id],s) }
                }
            }
            else -> Log.w(TAG, "⚠️ 未知 shape: ${shape.toList()}")
        }
        return list
    }

    // ════════════════════════════════════════════════════════
    // ✅ 旋转坐标变换（传感器坐标系 → 屏幕坐标系）
    // ════════════════════════════════════════════════════════
    private fun rotateNormalizedBox(box: RectF, rotation: Int): RectF = when (rotation) {
        90  -> RectF((1f - box.bottom).coerceIn(0f, 1f),
            (box.left).coerceIn(0f, 1f),
            (1f - box.top).coerceIn(0f, 1f),
            (box.right).coerceIn(0f, 1f))
        180 -> RectF(1f - box.right,  1f - box.bottom, 1f - box.left,  1f - box.top   )
        270 -> RectF(box.top,         1f - box.right,  box.bottom,     1f - box.left  )
        else -> box
    }

    // ════════════════════════════════════════════════════════
    // 检测结果 → TTS
    // ════════════════════════════════════════════════════════
    private fun handleDetections(dets: List<Detection>) {
        if (!ttsReady || dets.isEmpty()) return
        val now = System.currentTimeMillis()
        if (now - lastSpeakTime < SPEAK_INTERVAL_MS) return

        val top = dets
            .filter { it.labelName in dangerPriority }
            .maxByOrNull { (dangerPriority.size - dangerPriority.indexOf(it.labelName)) * it.score }
            ?: return

        val name = labelZh[top.labelName] ?: top.labelName
        val pos  = posHint(top.box)
        val msg  = "注意，${pos}有${name}"
        Log.d(TAG, "🔊 $msg  cx=${"%.2f".format((top.box.left+top.box.right)/2f)}")
        speak(msg)
        lastSpeakTime = now
    }

    private fun posHint(box: RectF): String {
        val cx = (box.left + box.right) / 2f
        return when {
            cx < 0.33f -> "左前方"
            cx > 0.66f -> "右前方"
            else       -> "正前方"
        }
    }

    // ════════════════════════════════════════════════════════
    // 模式切换
    // ════════════════════════════════════════════════════════
    private fun switchToDetection() {
        appModeState.value    = AppMode.DETECTION
        statusTextState.value = "障碍物检测运行中"
        speak("已启动障碍物检测")
    }

    private fun switchToIdle() {
        appModeState.value    = AppMode.IDLE
        detectionState.value  = emptyList()
        statusTextState.value = "等待语音指令"
    }

    // ════════════════════════════════════════════════════════
    // ✅ 修复语音识别 ERROR 9（INSUFFICIENT_PERMISSIONS）
    //    先检查麦克风权限 → 有权限才创建 SpeechRecognizer
    //    另外 TTS 播报和录音之间加 1.5s 延迟，避免录到自己的声音
    // ════════════════════════════════════════════════════════
    private fun startVoiceCommand() {
        if (!hasPerm(Manifest.permission.RECORD_AUDIO)) {
            isRequestingVoiceCommand = true
            statusTextState.value = "需要麦克风权限才能使用语音指令"
            Toast.makeText(this, "需要麦克风权限才能使用语音指令", Toast.LENGTH_SHORT).show()
            permissionLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
            return
        }
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            Toast.makeText(this, "设备不支持语音识别", Toast.LENGTH_SHORT).show()
            return
        }
        statusTextState.value = "准备聆听…"
        speak("请说出指令")
        mainHandler.postDelayed({ doStartListening() }, 1500) // 等 TTS 播完
    }

    private fun doStartListening() {
        speechRecognizer?.destroy()
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        speechRecognizer!!.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(p: Bundle?) { statusTextState.value = "🎤 聆听中…" }
            override fun onEndOfSpeech()              { statusTextState.value = "处理中…" }
            override fun onResults(r: Bundle?) {
                val text = r?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.firstOrNull() ?: ""
                Log.d(TAG, "🎤 识别: $text")
                statusTextState.value = "识别到：$text"
                handleVoiceCommand(text)
            }
            override fun onError(err: Int) {
                val desc = when (err) {
                    SpeechRecognizer.ERROR_NO_MATCH                 -> "未匹配到语音"
                    SpeechRecognizer.ERROR_SPEECH_TIMEOUT           -> "语音超时"
                    SpeechRecognizer.ERROR_AUDIO                    -> "录音错误"
                    SpeechRecognizer.ERROR_NETWORK                  -> "网络错误（语音识别需联网）"
                    SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "麦克风权限不足"
                    else -> "错误码 $err"
                }
                Log.e(TAG, "🎤 识别失败：$desc")
                statusTextState.value = "识别失败：$desc"

                // 如果是等待确认时失败，重置状态
                if (waitingForConfirm) {
                    waitingForConfirm = false
                    tempDestination = ""
                    speak("识别失败，" + desc + "。请重新告诉我要去哪里")
                    mainHandler.postDelayed({ startVoiceCommand() }, 2500)
                } else {
                    speak("识别失败，" + desc + "。请再说一次")
                    mainHandler.postDelayed({ startVoiceCommand() }, 2500)
                }
            }
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(v: Float) {}
            override fun onBufferReceived(b: ByteArray?) {}
            override fun onPartialResults(r: Bundle?) {}
            override fun onEvent(t: Int, p: Bundle?) {}
        })
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "zh-CN")
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }
        speechRecognizer!!.startListening(intent)
    }

    // ════════════════════════════════════════════════════════
    // 语音指令路由
    // ════════════════════════════════════════════════════════
    private var waitingForConfirm = false // 是否正在等待用户确认目的地
    private var tempDestination = "" // 临时存储识别到的目的地

    private fun handleVoiceCommand(cmd: String) {
        Log.d(TAG, "🎤 处理指令：$cmd, waitingForConfirm=$waitingForConfirm")

        // 如果正在等待确认
        if (waitingForConfirm) {
            if (cmd.contains("确认") || cmd.contains("是的") || cmd.contains("对") ||
                cmd.contains("好") || cmd.contains("开始") || cmd.contains("出发")) {
                // 用户确认，开始导航
                waitingForConfirm = false
                startNavigation(tempDestination)
            } else if (cmd.contains("取消") || cmd.contains("不对") || cmd.contains("错误") ||
                cmd.contains("不") || cmd.contains("重新")) {
                // 用户否认，重新询问
                waitingForConfirm = false
                tempDestination = ""
                speak("好的，请重新告诉我要去哪里")
                mainHandler.postDelayed({ startVoiceCommand() }, 2000)
            } else {
                // 用户说了其他内容，再次询问
                speak("请问您是要去" + tempDestination + "吗？请说确认或取消")
                mainHandler.postDelayed({ startVoiceCommand() }, 2000)
            }
            return
        }

        // 正常处理语音指令
        when {
            cmd.contains("去") || cmd.contains("导航") || cmd.contains("怎么走") -> {
                val dest = cmd.replace(Regex("(帮我|我要|我想|带我|导航到|去|怎么走)"), "").trim()
                if (dest.isNotEmpty()) {
                    // 询问用户确认
                    tempDestination = dest
                    waitingForConfirm = true
                    speak("请问您是要去" + dest + "吗？请说确认或取消")
                    // 2 秒后自动开始语音识别，让用户可以直接回答
                    mainHandler.postDelayed({
                        if (waitingForConfirm) {
                            startVoiceCommand()
                        }
                    }, 2000)
                } else {
                    speak("请告诉我目的地")
                    mainHandler.postDelayed({ startVoiceCommand() }, 2000)
                }
            }
            cmd.contains("识别") || cmd.contains("周围") ||
                    cmd.contains("前面") || cmd.contains("障碍") -> switchToDetection()
            cmd.contains("停止") || cmd.contains("退出") -> switchToIdle()
            else -> {
                speak("没有理解，可以说：去某地，或者识别周围环境")
                mainHandler.postDelayed({ startVoiceCommand() }, 2500)
            }
        }
    }

    /**
     * 高德地图导航（预留接口）
     * 集成步骤：
     *  1. build.gradle.kts：implementation("com.amap.api:navi-3dmap:latest.release")
     *  2. AndroidManifest.xml：
     *     <meta-data android:name="com.amap.api.v2.apikey" android:value="YOUR_AMAP_KEY"/>
     *  3. 替换下方 TODO 为 AMapNavi 调用
     */
    private fun startNavigation(destination: String) {
        appModeState.value    = AppMode.NAVIGATION
        statusTextState.value = "正在规划「$destination」的路线…"
        speak("好的，正在为您规划前往${destination}的路线")
        Log.d(TAG, "高德导航 → $destination")
        // 启动导航活动
        val intent = Intent(this, NaviActivity::class.java)
        intent.putExtra("dest", destination)
        startActivity(intent)
        // 移除自动切换回检测模式的代码，让导航活动正常运行
    }

    // ════════════════════════════════════════════════════════
    // TTS 工具
    // ════════════════════════════════════════════════════════
    private fun speak(text: String) {
        if (!ttsReady) return
        Log.d(TAG, "🔊 $text")
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, UUID.randomUUID().toString())
    }

    // ════════════════════════════════════════════════════════
    // UI 辅助
    // ════════════════════════════════════════════════════════
    private fun modeLabel(mode: AppMode) = when (mode) {
        AppMode.IDLE       -> "⏳ 待机"
        AppMode.NAVIGATION -> "🗺 导航中"
        AppMode.DETECTION  -> "👁 障碍检测"
        AppMode.AI_SCENE   -> "🤖 AI识别"
    }

    private fun modeColor(mode: AppMode) = when (mode) {
        AppMode.IDLE       -> androidx.compose.ui.graphics.Color(0xFF607D8B)
        AppMode.NAVIGATION -> androidx.compose.ui.graphics.Color(0xFF1976D2)
        AppMode.DETECTION  -> androidx.compose.ui.graphics.Color(0xFFD32F2F)
        AppMode.AI_SCENE   -> androidx.compose.ui.graphics.Color(0xFF7B1FA2)
    }

    // ════════════════════════════════════════════════════════
    // 常量
    // ════════════════════════════════════════════════════════
    companion object {
        private const val TAG               = "VisionPath"
        private const val CONF_THRESHOLD    = 0.30f
        private const val SPEAK_INTERVAL_MS = 3000L
    }
}