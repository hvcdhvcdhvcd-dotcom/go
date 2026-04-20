package com.example.myapplication

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.Button
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.amap.api.maps.MapsInitializer
import com.amap.api.navi.AMapNavi
import com.amap.api.navi.SimpleNaviListener
import com.amap.api.navi.enums.NaviType
import com.amap.api.navi.model.AMapCalcRouteResult
import com.amap.api.navi.model.AMapNaviLocation
import com.amap.api.navi.model.NaviInfo
import com.amap.api.navi.model.NaviLatLng
import com.amap.api.services.core.AMapException
import com.amap.api.services.geocoder.GeocodeResult
import com.amap.api.services.geocoder.GeocodeSearch
import com.amap.api.services.geocoder.GeocodeQuery
import com.amap.api.services.geocoder.RegeocodeResult
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.util.*
import java.util.concurrent.Executors

// ONNX模型相关常量
private const val CONF_THRESHOLD = 0.3f // 置信度阈值
private const val SPEAK_INTERVAL_MS = 3000L // 语音播报间隔

// 语音消息优先级
enum class MessagePriority {
    EMERGENCY,  // 紧急障碍
    IMPORTANT,  // 重要导航
    NORMAL,     // 普通障碍
    INFO        // 普通导航
}

// 语音消息类型
enum class MessageType {
    NAVIGATION, OBSTACLE, SYSTEM
}

// 语音消息数据结构
data class VoiceMessage(
    val priority: MessagePriority,
    val content: String,
    val type: MessageType,
    val timestamp: Long = System.currentTimeMillis()
)

// 障碍物检测结果
data class ObstacleDetection(
    val label: String,
    val confidence: Float,
    val bbox: android.graphics.RectF,
    val distance: Float = 0f, // 估算距离（米）
    val dangerLevel: DangerLevel = DangerLevel.LOW
)

// 危险等级
enum class DangerLevel {
    LOW,        // 低风险
    MEDIUM,     // 中等风险
    HIGH,       // 高风险
    CRITICAL    // 紧急风险
}

// 障碍物类型优先级
private val obstaclePriority = listOf(
    "car", "bus", "truck", "motorcycle", "bicycle", "person",
    "traffic light", "stop sign", "chair", "couch", "dining table", "bottle", "laptop"
)

// 障碍物中文名称映射
private val obstacleLabelZh = mapOf(
    "person" to "行人", "bicycle" to "自行车", "car" to "汽车", "motorcycle" to "摩托车",
    "bus" to "公共汽车", "truck" to "卡车", "traffic light" to "红绿灯", "stop sign" to "停止标志",
    "bench" to "长椅", "chair" to "椅子", "couch" to "沙发", "dining table" to "桌子",
    "bottle" to "瓶子", "cup" to "杯子", "cat" to "猫", "dog" to "狗",
    "laptop" to "笔记本电脑", "mouse" to "鼠标", "keyboard" to "键盘", "cell phone" to "手机",
    "tv" to "电视", "bed" to "床", "toilet" to "马桶", "sink" to "水槽", "refrigerator" to "冰箱",
    "backpack" to "背包", "umbrella" to "雨伞", "book" to "书", "clock" to "时钟",
    "potted plant" to "盆栽", "scissors" to "剪刀", "fire hydrant" to "消防栓",
    "bird" to "鸟", "horse" to "马", "cow" to "牛"
)

class IntegratedNaviActivity : Activity(), TextToSpeech.OnInitListener,
    GeocodeSearch.OnGeocodeSearchListener, RecognitionListener {

    // ==================== 导航相关变量 ====================
    private lateinit var aMapNavi: AMapNavi
    private var destination: String = ""
    private var currentLocation: NaviLatLng? = null
    private var destinationLatLng: NaviLatLng? = null
    private var geocodeSearch: GeocodeSearch? = null
    private var pendingGeocodeCallback: ((NaviLatLng?) -> Unit)? = null

    // ==================== 障碍检测相关变量 ====================
    private lateinit var previewView: PreviewView
    private var cameraProvider: ProcessCameraProvider? = null
    private var isCameraInitialized = false
    private val detectionHandler = Handler(Looper.getMainLooper())
    private var lastDetectionTime = 0L
    private val detectionInterval = 500L // 检测间隔500ms

    // ONNX模型相关
    private lateinit var ortEnv: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private var modelReady = false
    private var lastSpeakTime = 0L
    private var currentRotationDegrees = 0

    // 障碍检测结果
    private val currentDetections = mutableListOf<ObstacleDetection>()

    // ==================== 语音相关变量 ====================
    private lateinit var tts: TextToSpeech
    private var ttsReady = false
    private var speechRecognizer: SpeechRecognizer? = null
    private var isListening = false
    private var waitingForDestination = false
    private var tempDestination = "" // 临时存储识别到的目的地

    // ==================== 语音队列管理 ====================
    private val voiceQueue = LinkedList<VoiceMessage>()
    private var isSpeaking = false
    private val voiceHandler = Handler(Looper.getMainLooper())
    private var currentSpeakingMessage: VoiceMessage? = null

    // ==================== UI 组件 ====================
    private lateinit var tvInstruction: TextView
    private lateinit var tvDistance: TextView
    private lateinit var tvTime: TextView
    private lateinit var tvNext: TextView
    private lateinit var tvDestination: TextView
    private lateinit var btnVoiceCommand: Button
    private lateinit var btnBack: ImageButton

    // ==================== 位置服务 ====================
    private lateinit var locationManager: LocationManager
    private var locationListener: LocationListener? = null

    // ==================== 权限请求 ====================
    private fun checkAndRequestPermissions() {
        val neededPermissions = mutableListOf<String>()

        if (!hasPermission(Manifest.permission.CAMERA)) {
            neededPermissions.add(Manifest.permission.CAMERA)
        }
        if (!hasPermission(Manifest.permission.RECORD_AUDIO)) {
            neededPermissions.add(Manifest.permission.RECORD_AUDIO)
        }
        if (!hasPermission(Manifest.permission.ACCESS_FINE_LOCATION)) {
            neededPermissions.add(Manifest.permission.ACCESS_FINE_LOCATION)
        }

        if (neededPermissions.isEmpty()) {
            initializeAllComponents()
        } else {
            // 简化权限处理，直接请求
            ActivityCompat.requestPermissions(this, neededPermissions.toTypedArray(), 100)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100) {
            val allGranted = grantResults.all { it == PackageManager.PERMISSION_GRANTED }
            if (allGranted) {
                initializeAllComponents()
            } else {
                Toast.makeText(this, "需要所有权限才能使用完整功能", Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_integrated_navi)

        // 初始化UI组件
        initializeUI()

        // 检查权限
        checkAndRequestPermissions()
    }

    private fun initializeUI() {
        tvInstruction = findViewById(R.id.tvInstruction)
        tvDistance = findViewById(R.id.tvDistance)
        tvTime = findViewById(R.id.tvTime)
        tvNext = findViewById(R.id.tvNext)
        tvDestination = findViewById(R.id.tvDestination)
        btnVoiceCommand = findViewById(R.id.btnVoiceCommand)
        btnBack = findViewById(R.id.btnBack)
        previewView = findViewById(R.id.previewView)

        // 设置按钮点击事件
        btnVoiceCommand.setOnClickListener {
            if (!isListening) {
                startVoiceCommand()
            } else {
                stopVoiceCommand()
            }
        }

        btnBack.setOnClickListener {
            finish()
        }

        // 初始化UI状态
        updateInstruction("请说出目的地")
        updateDistance("-")
        updateTime("-")
        updateNext("等待指令")
        updateDestination("未设置")
    }

    private fun checkPermissions() {
        val neededPermissions = mutableListOf<String>()

        if (!hasPermission(Manifest.permission.CAMERA)) {
            neededPermissions.add(Manifest.permission.CAMERA)
        }
        if (!hasPermission(Manifest.permission.RECORD_AUDIO)) {
            neededPermissions.add(Manifest.permission.RECORD_AUDIO)
        }
        if (!hasPermission(Manifest.permission.ACCESS_FINE_LOCATION)) {
            neededPermissions.add(Manifest.permission.ACCESS_FINE_LOCATION)
        }

        if (neededPermissions.isEmpty()) {
            initializeAllComponents()
        } else {
            ActivityCompat.requestPermissions(this, neededPermissions.toTypedArray(), 100)
        }
    }

    private fun hasPermission(permission: String): Boolean {
        return ActivityCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
    }

    private fun initializeAllComponents() {
        // 初始化TTS
        tts = TextToSpeech(this, this)

        // 初始化ONNX模型
        initializeONNXModel()

        // 初始化高德地图导航
        initializeAMapNavi()

        // 初始化相机和障碍检测
        initializeCamera()

        // 初始化位置服务
        initializeLocationService()

        // 初始化语音识别
        initializeSpeechRecognizer()

        // 设置高德地图隐私合规
        setAMapPrivacySettings()
    }

    // ==================== 导航监听器 ====================
    private val naviListener = object : SimpleNaviListener() {
        override fun onInitNaviSuccess() {
            Log.d("IntegratedNavi", "✅ 高德导航初始化成功")
            addVoiceMessage(VoiceMessage(MessagePriority.INFO, "导航系统已就绪", MessageType.SYSTEM))
        }

        override fun onCalculateRouteSuccess(p0: AMapCalcRouteResult?) {
            Log.d("IntegratedNavi", "✅ 路线规划成功")
            aMapNavi.startNavi(NaviType.GPS)
            addVoiceMessage(VoiceMessage(MessagePriority.IMPORTANT, "路线规划完成，开始导航", MessageType.NAVIGATION))
        }

        override fun onCalculateRouteFailure(p0: AMapCalcRouteResult?) {
            Log.e("IntegratedNavi", "❌ 路线规划失败")
            addVoiceMessage(VoiceMessage(MessagePriority.IMPORTANT, "路线规划失败，请重试", MessageType.SYSTEM))
        }

        override fun onArriveDestination() {
            Log.d("IntegratedNavi", "✅ 到达目的地")
            addVoiceMessage(VoiceMessage(MessagePriority.IMPORTANT, "已到达目的地", MessageType.NAVIGATION))
            updateInstruction("已到达目的地")
        }

        override fun onNaviInfoUpdate(naviInfo: NaviInfo?) {
            naviInfo?.let { info ->
                updateNavigationInfo(info)
            }
        }

        override fun onLocationChange(location: AMapNaviLocation?) {
            location?.let { loc ->
                currentLocation = NaviLatLng(loc.coord.latitude, loc.coord.longitude)
            }
        }
    }

    private fun initializeAMapNavi() {
        try {
            aMapNavi = AMapNavi.getInstance(applicationContext)
            aMapNavi.addAMapNaviListener(naviListener)
        } catch (e: Exception) {
            Log.e("IntegratedNavi", "❌ 高德导航初始化失败: ${e.message}")
        }
    }

    private fun initializeCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(Runnable {
            try {
                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases()
            } catch (e: Exception) {
                Log.e("IntegratedNavi", "❌ 相机初始化失败: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: return

        try {
            // 解绑所有用例
            cameraProvider.unbindAll()

            // 预览用例
            val preview = androidx.camera.core.Preview.Builder()
                .setTargetResolution(Size(640, 480))
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            // 图像分析用例（障碍检测）
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 640))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                        processImageForObstacleDetection(imageProxy)
                    }
                }

            // 绑定到生命周期
            cameraProvider.bindToLifecycle(
                this as androidx.lifecycle.LifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalysis
            )

            isCameraInitialized = true
            Log.d("IntegratedNavi", "✅ 相机绑定成功")
        } catch (e: Exception) {
            Log.e("IntegratedNavi", "❌ 相机绑定失败: ${e.message}")
        }
    }

    private fun processImageForObstacleDetection(imageProxy: ImageProxy) {
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastDetectionTime < detectionInterval) {
            imageProxy.close()
            return
        }
        lastDetectionTime = currentTime

        try {
            if (modelReady) {
                // 处理图像并检测障碍物
                val detections = processFrameWithONNX(imageProxy)
                handleObstacleDetections(detections)
            } else {
                // 模型未就绪，使用模拟检测
                simulateObstacleDetection()
            }
        } catch (e: Exception) {
            Log.e("IntegratedNavi", "❌ 障碍检测处理失败: ${e.message}")
        } finally {
            imageProxy.close()
        }
    }

    private fun processFrameWithONNX(proxy: ImageProxy): List<ObstacleDetection> {
        val detections = mutableListOf<ObstacleDetection>()

        try {
            // 图像预处理
            val raw = imageProxyToBitmap(proxy) ?: return emptyList()
            val processed = letterboxBitmap(raw, 640)
            raw.recycle()

            // 转换为模型输入格式
            val inputData = bitmapToFloatBuffer(processed)
            processed.recycle()

            // ONNX推理
            val inputName = ortSession.inputNames.iterator().next()
            val tensor = OnnxTensor.createTensor(
                ortEnv, java.nio.FloatBuffer.wrap(inputData), longArrayOf(1, 3, 640, 640)
            )

            tensor.use {
                val output = ortSession.run(Collections.singletonMap(inputName, tensor))
                output.use { o ->
                    val outputTensor = o[0] as OnnxTensor
                    val outputArray = FloatArray(outputTensor.floatBuffer.remaining()).also {
                        outputTensor.floatBuffer.get(it)
                    }

                    // 解析检测结果
                    val rawDetections = parseONNXDetections(outputArray, outputTensor.info.shape)
                    detections.addAll(enhanceDetectionsWithDistanceAndDanger(rawDetections))
                }
            }
        } catch (e: Exception) {
            Log.e("IntegratedNavi", "❌ ONNX推理失败: ${e.message}")
        }

        return detections
    }

    private fun parseONNXDetections(arr: FloatArray, shape: LongArray): List<ObstacleDetection> {
        val detections = mutableListOf<ObstacleDetection>()

        // 解析ONNX输出格式 [batch_size, num_detections, 6]
        // 6个值: [x1, y1, x2, y2, confidence, class_id]
        if (shape.size == 3 && shape[2].toInt() == 6) {
            val numDetections = shape[1].toInt()
            for (i in 0 until numDetections) {
                val offset = i * 6
                val confidence = arr[offset + 4]

                if (confidence > CONF_THRESHOLD) {
                    val classId = arr[offset + 5].toInt()
                    val label = getLabelForClassId(classId)
                    val bbox = android.graphics.RectF(
                        arr[offset], arr[offset + 1],
                        arr[offset + 2], arr[offset + 3]
                    )

                    detections.add(ObstacleDetection(label, confidence, bbox))
                }
            }
        }

        return detections
    }

    private fun getLabelForClassId(classId: Int): String {
        // 简化的类别映射，实际应根据YOLO模型训练时的类别顺序
        val labels = listOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee"
        )
        return labels.getOrElse(classId) { "object_$classId" }
    }

    private fun enhanceDetectionsWithDistanceAndDanger(detections: List<ObstacleDetection>): List<ObstacleDetection> {
        return detections.map { detection ->
            // 估算距离（基于物体大小和位置）
            val distance = estimateDistance(detection.bbox, detection.label)

            // 判断危险等级
            val dangerLevel = calculateDangerLevel(detection.label, distance, detection.confidence)

            detection.copy(distance = distance, dangerLevel = dangerLevel)
        }
    }

    private fun estimateDistance(bbox: android.graphics.RectF, label: String): Float {
        // 基于物体在图像中的大小估算距离
        val bboxArea = (bbox.right - bbox.left) * (bbox.bottom - bbox.top)

        // 不同物体类型的基准大小和距离估算
        return when (label) {
            "person" -> 10f / bboxArea.coerceAtLeast(0.01f)
            "car", "bus", "truck" -> 20f / bboxArea.coerceAtLeast(0.01f)
            "bicycle", "motorcycle" -> 15f / bboxArea.coerceAtLeast(0.01f)
            else -> 5f / bboxArea.coerceAtLeast(0.01f)
        }.coerceIn(0.5f, 50f) // 限制距离范围
    }

    private fun calculateDangerLevel(label: String, distance: Float, confidence: Float): DangerLevel {
        // 基于物体类型、距离和置信度计算危险等级
        val baseRisk = when (label) {
            "car", "bus", "truck", "motorcycle" -> 3
            "person", "bicycle" -> 2
            "traffic light", "stop sign" -> 1
            else -> 0
        }

        val distanceRisk = when {
            distance < 3f -> 3
            distance < 5f -> 2
            distance < 10f -> 1
            else -> 0
        }

        val confidenceRisk = if (confidence > 0.8f) 1 else 0

        val totalRisk = baseRisk + distanceRisk + confidenceRisk

        return when (totalRisk) {
            in 0..1 -> DangerLevel.LOW
            in 2..3 -> DangerLevel.MEDIUM
            in 4..5 -> DangerLevel.HIGH
            else -> DangerLevel.CRITICAL
        }
    }

    private fun handleObstacleDetections(detections: List<ObstacleDetection>) {
        currentDetections.clear()
        currentDetections.addAll(detections)

        // 过滤误检
        val filteredDetections = filterFalseDetections(detections)

        // 检查是否有需要紧急响应的障碍物
        val criticalDetections = filteredDetections.filter { it.dangerLevel == DangerLevel.CRITICAL }
        if (criticalDetections.isNotEmpty()) {
            // 触发紧急响应
            handleEmergencySituation(criticalDetections.first())
        } else {
            // 正常处理障碍物检测
            generateObstacleVoiceMessages(filteredDetections)
        }
    }

    private fun filterFalseDetections(detections: List<ObstacleDetection>): List<ObstacleDetection> {
        return detections.filter { detection ->
            // 置信度过滤
            detection.confidence > CONF_THRESHOLD &&
                    // 物体大小过滤（避免过小的误检）
                    (detection.bbox.right - detection.bbox.left) > 0.05f &&
                    (detection.bbox.bottom - detection.bbox.top) > 0.05f &&
                    // 距离过滤（避免过远的物体）
                    detection.distance < 20f
        }
    }

    private fun generateObstacleVoiceMessages(detections: List<ObstacleDetection>) {
        val now = System.currentTimeMillis()
        if (now - lastSpeakTime < SPEAK_INTERVAL_MS) return

        // 按危险等级排序，优先处理高风险障碍
        val sortedDetections = detections.sortedByDescending { it.dangerLevel.ordinal }

        sortedDetections.firstOrNull()?.let { topDetection ->
            val message = createObstacleMessage(topDetection)
            val priority = when (topDetection.dangerLevel) {
                DangerLevel.CRITICAL -> MessagePriority.EMERGENCY
                DangerLevel.HIGH -> MessagePriority.IMPORTANT
                DangerLevel.MEDIUM -> MessagePriority.NORMAL
                DangerLevel.LOW -> MessagePriority.INFO
            }

            addVoiceMessage(VoiceMessage(priority, message, MessageType.OBSTACLE))
            lastSpeakTime = now
        }
    }

    private fun createObstacleMessage(detection: ObstacleDetection): String {
        val chineseName = obstacleLabelZh[detection.label] ?: detection.label
        val position = getObstaclePosition(detection.bbox)
        val distanceText = when {
            detection.distance < 3f -> "很近"
            detection.distance < 5f -> "较近"
            detection.distance < 10f -> "前方"
            else -> "较远"
        }

        return when (detection.dangerLevel) {
            DangerLevel.CRITICAL -> "紧急！${position}有${chineseName}，距离${distanceText}，请立即避让！"
            DangerLevel.HIGH -> "注意！${position}有${chineseName}，距离${distanceText}，请小心"
            DangerLevel.MEDIUM -> "${position}有${chineseName}，距离${distanceText}"
            DangerLevel.LOW -> "${position}检测到${chineseName}"
        }
    }

    private fun getObstaclePosition(bbox: android.graphics.RectF): String {
        val centerX = (bbox.left + bbox.right) / 2f
        return when {
            centerX < 0.33f -> "左前方"
            centerX > 0.66f -> "右前方"
            else -> "正前方"
        }
    }

    private fun simulateObstacleDetection() {
        // 模拟检测到障碍物
        val random = Random()
        if (random.nextFloat() < 0.3) { // 30%概率检测到障碍
            val obstacles = listOf("前方有行人", "左侧有车辆", "右侧有障碍物", "前方有红绿灯")
            val obstacle = obstacles[random.nextInt(obstacles.size)]

            detectionHandler.post {
                addVoiceMessage(VoiceMessage(MessagePriority.NORMAL, obstacle, MessageType.OBSTACLE))
            }
        }
    }

    // ==================== 图像处理辅助函数 ====================
    private fun imageProxyToBitmap(proxy: ImageProxy): android.graphics.Bitmap? = runCatching {
        val plane = proxy.planes[0]
        val ps = plane.pixelStride
        val rs = plane.rowStride
        val pad = rs - ps * proxy.width
        val bmp = android.graphics.Bitmap.createBitmap(proxy.width + pad / ps, proxy.height, android.graphics.Bitmap.Config.ARGB_8888)
        bmp.copyPixelsFromBuffer(plane.buffer)
        if (pad > 0) {
            val crop = android.graphics.Bitmap.createBitmap(bmp, 0, 0, proxy.width, proxy.height)
            bmp.recycle(); crop
        } else bmp
    }.getOrElse { e -> Log.e("IntegratedNavi", "❌ Bitmap转换失败: ${e.message}"); null }

    private fun letterboxBitmap(src: android.graphics.Bitmap, target: Int): android.graphics.Bitmap {
        val scale = minOf(target.toFloat() / src.width, target.toFloat() / src.height)
        val sw = (src.width * scale).toInt()
        val sh = (src.height * scale).toInt()
        val padL = (target - sw) / 2
        val padT = (target - sh) / 2
        val dst = android.graphics.Bitmap.createBitmap(target, target, android.graphics.Bitmap.Config.ARGB_8888)
        dst.eraseColor(android.graphics.Color.argb(255, 114, 114, 114))
        val c = android.graphics.Canvas(dst)
        val scaled = android.graphics.Bitmap.createScaledBitmap(src, sw, sh, true)
        c.drawBitmap(scaled, padL.toFloat(), padT.toFloat(), null)
        if (scaled !== src) scaled.recycle()
        return dst
    }

    private fun bitmapToFloatBuffer(bmp: android.graphics.Bitmap): FloatArray {
        val n = 640 * 640
        val buf = FloatArray(3 * n)
        val px = IntArray(n)
        bmp.getPixels(px, 0, 640, 0, 0, 640, 640)
        for (i in 0 until n) {
            val c = px[i]
            buf[i] = (c shr 16 and 0xFF) / 255f
            buf[n + i] = (c shr 8 and 0xFF) / 255f
            buf[2 * n + i] = (c and 0xFF) / 255f
        }
        return buf
    }

    private fun initializeLocationService() {
        locationManager = getSystemService(LOCATION_SERVICE) as LocationManager

        locationListener = object : LocationListener {
            override fun onLocationChanged(location: Location) {
                currentLocation = NaviLatLng(location.latitude, location.longitude)
                Log.d("IntegratedNavi", "📍 位置更新: ${location.latitude}, ${location.longitude}")
            }

            override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
            override fun onProviderEnabled(provider: String) {}
            override fun onProviderDisabled(provider: String) {}
        }

        try {
            if (hasPermission(Manifest.permission.ACCESS_FINE_LOCATION)) {
                locationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER,
                    1000L,
                    1f,
                    locationListener!!
                )
            }
        } catch (e: SecurityException) {
            Log.e("IntegratedNavi", "❌ 位置服务权限错误: ${e.message}")
        }
    }

    private fun initializeSpeechRecognizer() {
        try {
            // 检查语音识别是否可用
            if (!SpeechRecognizer.isRecognitionAvailable(this)) {
                Log.e("IntegratedNavi", "❌ 语音识别不可用")
                Toast.makeText(this, "设备不支持语音识别", Toast.LENGTH_LONG).show()
                return
            }

            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
            speechRecognizer?.setRecognitionListener(this)
            Log.d("IntegratedNavi", "✅ 语音识别初始化成功")
        } catch (e: Exception) {
            Log.e("IntegratedNavi", "❌ 语音识别初始化失败: ${e.message}")
            Toast.makeText(this, "语音识别初始化失败", Toast.LENGTH_LONG).show()
        }
    }

    private fun setAMapPrivacySettings() {
        try {
            val method = MapsInitializer::class.java.getDeclaredMethod("updatePrivacyShow", android.content.Context::class.java, Boolean::class.java, Boolean::class.java)
            method.isAccessible = true
            method.invoke(null, this, true, true)

            val method2 = MapsInitializer::class.java.getDeclaredMethod("updatePrivacyAgree", android.content.Context::class.java, Boolean::class.java)
            method2.isAccessible = true
            method2.invoke(null, this, true)

            Log.d("IntegratedNavi", "✅ 高德隐私设置完成")
        } catch (e: Exception) {
            Log.e("IntegratedNavi", "❌ 高德隐私设置失败: ${e.message}")
        }
    }

    // ==================== ONNX模型初始化 ====================
    private fun initializeONNXModel() {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val bytes = assets.open("yolov10n.onnx").readBytes()
            ortSession = ortEnv.createSession(bytes, OrtSession.SessionOptions())
            modelReady = true
            Log.d("IntegratedNavi", "✅ ONNX模型加载成功")
        } catch (e: Exception) {
            Log.e("IntegratedNavi", "❌ ONNX模型加载失败: ${e.message}")
            addVoiceMessage(VoiceMessage(MessagePriority.INFO, "障碍检测模型加载失败", MessageType.SYSTEM))
        }
    }

    // ==================== 智能语音调度系统 ====================
    private fun addVoiceMessage(message: VoiceMessage) {
        // 检查是否已有相同类型的消息在队列中，避免重复
        if (shouldFilterDuplicateMessage(message)) {
            Log.d("IntegratedNavi", "🔇 过滤重复消息: ${message.content}")
            return
        }

        voiceQueue.add(message)
        processVoiceQueue()
    }

    private fun shouldFilterDuplicateMessage(newMessage: VoiceMessage): Boolean {
        // 修复问题二：只按内容完全相同才过滤，不再按类型（OBSTACLE）整体拦截
        // 原逻辑中 "同类型就过滤" 会把所有障碍消息挡掉，导致导航时无法播报障碍
        val recentMessages = voiceQueue.filter {
            System.currentTimeMillis() - it.timestamp < 5000
        }
        return recentMessages.any { it.content == newMessage.content }
    }

    private fun processVoiceQueue() {
        if (isSpeaking || voiceQueue.isEmpty()) return

        // 智能优先级排序：紧急消息优先，同时考虑消息类型和时间
        val sortedQueue = voiceQueue.sortedWith(compareBy(
            { -it.priority.ordinal }, // 优先级降序
            { it.timestamp }          // 时间戳升序（同优先级按时间）
        ))

        voiceQueue.clear()
        voiceQueue.addAll(sortedQueue)

        val message = voiceQueue.poll() ?: return
        currentSpeakingMessage = message
        isSpeaking = true

        // 根据消息类型和优先级调整语音参数
        adjustTTSForMessage(message)

        // 修复问题一：speak() 内部使用带 utteranceId 的调用，
        // TTS 播完后由 UtteranceProgressListener.onDone 回调 processVoiceQueue，
        // 不再依赖延迟估算，彻底消除时序错位
        speakQueued(message.content)

        // 更新UI显示当前播报内容
        updateCurrentSpeaking(message.content, message.type)
    }

    // 专门供队列调用的 speak，使用 QUEUE_ADD 避免截断自身队列内的消息
    private fun speakQueued(text: String) {
        if (ttsReady) {
            tts.speak(text, TextToSpeech.QUEUE_ADD, null, "queued_${System.currentTimeMillis()}")
        } else {
            // TTS 未就绪时直接重置标志，防止队列卡死
            voiceHandler.postDelayed({
                isSpeaking = false
                processVoiceQueue()
            }, 500)
        }
    }

    private fun adjustTTSForMessage(message: VoiceMessage) {
        if (!ttsReady) return

        when (message.priority) {
            MessagePriority.EMERGENCY -> {
                tts.setSpeechRate(1.2f) // 加快语速
                tts.setPitch(1.3f)     // 提高音调
            }
            MessagePriority.IMPORTANT -> {
                tts.setSpeechRate(1.0f)
                tts.setPitch(1.1f)
            }
            MessagePriority.NORMAL -> {
                tts.setSpeechRate(0.9f)
                tts.setPitch(1.0f)
            }
            MessagePriority.INFO -> {
                tts.setSpeechRate(0.8f) // 减慢语速
                tts.setPitch(0.9f)     // 降低音调
            }
        }
    }

    private fun calculateSmartDelay(message: VoiceMessage): Long {
        val baseDuration = calculateSpeechDuration(message.content)

        // 根据消息类型和优先级调整延迟
        return when (message.priority) {
            MessagePriority.EMERGENCY -> baseDuration + 500L // 紧急消息后快速响应
            MessagePriority.IMPORTANT -> baseDuration + 1000L
            MessagePriority.NORMAL -> baseDuration + 1500L
            MessagePriority.INFO -> baseDuration + 2000L // 普通消息后稍作停顿
        }
    }

    private fun calculateSpeechDuration(text: String): Long {
        // 更精确的语音时长估算
        val chineseChars = text.count { it.toString().matches("\\p{IsHan}".toRegex()) }
        val otherChars = text.length - chineseChars

        // 考虑标点符号的停顿
        val punctuationCount = text.count { it in listOf('，', '。', '！', '？', '；') }

        return (chineseChars * 300L + otherChars * 100L + punctuationCount * 200L).coerceAtLeast(800L)
    }

    private fun updateCurrentSpeaking(text: String, type: MessageType) {
        // 在UI上显示当前播报内容
        val prefix = when (type) {
            MessageType.NAVIGATION -> "🚗 "
            MessageType.OBSTACLE -> "⚠️ "
            MessageType.SYSTEM -> "🔊 "
        }

        // 更新UI状态指示器
        runOnUiThread {
            findViewById<android.widget.TextView>(R.id.tvSpeakingStatus).apply {
                this.text = "$prefix$text"
                visibility = android.view.View.VISIBLE
            }
        }

        Log.d("IntegratedNavi", "$prefix$text")
    }

    // ==================== 紧急情况快速响应机制 ====================
    fun handleEmergencySituation(detection: ObstacleDetection) {
        // 修复问题二：立即打断并重置 isSpeaking，防止队列卡住
        tts.stop()
        isSpeaking = false
        currentSpeakingMessage = null

        // 清空队列，只保留紧急消息
        voiceQueue.clear()

        val emergencyMessage = createEmergencyMessage(detection)
        addVoiceMessage(VoiceMessage(MessagePriority.EMERGENCY, emergencyMessage, MessageType.OBSTACLE))

        triggerEmergencyAlert()
    }

    private fun createEmergencyMessage(detection: ObstacleDetection): String {
        val chineseName = obstacleLabelZh[detection.label] ?: detection.label
        val position = getObstaclePosition(detection.bbox)

        return when (detection.dangerLevel) {
            DangerLevel.CRITICAL -> "紧急！${position}有${chineseName}，距离很近，请立即停止前进！"
            DangerLevel.HIGH -> "危险！${position}有${chineseName}，请立即避让！"
            else -> "注意！${position}有${chineseName}，请小心"
        }
    }

    private fun triggerEmergencyAlert() {
        // 触发设备振动（兼容API级别）
        try {
            val vibrator = getSystemService(android.content.Context.VIBRATOR_SERVICE) as android.os.Vibrator
            if (vibrator.hasVibrator()) {
                val pattern = longArrayOf(0, 500, 200, 500) // 振动模式
                // 使用兼容的振动API
                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                    vibrator.vibrate(android.os.VibrationEffect.createWaveform(pattern, 0))
                } else {
                    @Suppress("DEPRECATION")
                    vibrator.vibrate(pattern, -1)
                }
            }
        } catch (e: Exception) {
            Log.e("IntegratedNavi", "❌ 振动失败: ${e.message}")
        }

        // 可以添加闪光灯等其他紧急提示
    }

    // ==================== 语音识别相关 ====================
    private fun startVoiceCommand() {
        if (!hasPermission(Manifest.permission.RECORD_AUDIO)) {
            Toast.makeText(this, "需要麦克风权限才能使用语音指令", Toast.LENGTH_SHORT).show()
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), 101)
            return
        }

        if (!ttsReady) {
            Toast.makeText(this, "语音系统未就绪", Toast.LENGTH_SHORT).show()
            return
        }

        if (isListening) {
            Toast.makeText(this, "正在聆听中，请稍候", Toast.LENGTH_SHORT).show()
            return
        }

        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            Toast.makeText(this, "设备不支持语音识别", Toast.LENGTH_SHORT).show()
            return
        }

        val prompt = if (waitingForDestination) {
            "请说确认开始导航，或重新设置目的地"
        } else {
            "请说出目的地，如：武汉大学、光谷广场、汉口火车站"
        }

        // 修复问题一：先停止当前 TTS 播报，然后直接 speak prompt，
        // 由 UtteranceProgressListener.onDone 完成后再启动识别，
        // 消除"播完就没反应"问题（原因是 isSpeaking 延迟估算完成后识别器状态不对）
        tts.stop()
        isSpeaking = false

        // 单独播报提示语，播完后 onDone 会调 processVoiceQueue，
        // 但这里我们绕过队列直接用一次性监听启动识别
        tts.speak(prompt, TextToSpeech.QUEUE_FLUSH, null, "voice_prompt")

        // 给 TTS 足够时间播完提示语后再启动识别
        val estimatedMs = (prompt.length * 250L).coerceAtLeast(1500L)
        voiceHandler.postDelayed({
            if (!isListening) {
                doStartListening()
            }
        }, estimatedMs)
    }

    private fun doStartListening() {
        try {
            // 修复问题一：只在未监听状态下重建识别器，避免多次 destroy 导致状态混乱
            if (speechRecognizer == null) {
                speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
                speechRecognizer?.setRecognitionListener(this)
            }

            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, "zh-CN")
                putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            }

            speechRecognizer?.startListening(intent)
            isListening = true
            btnVoiceCommand.text = "正在聆听..."

            Log.d("IntegratedNavi", "🎤 启动语音识别")

        } catch (e: Exception) {
            Log.e("IntegratedNavi", "❌ 启动语音识别失败: ${e.message}")
            isListening = false
            btnVoiceCommand.text = "语音指令"
            Toast.makeText(this, "语音识别启动失败，请重试", Toast.LENGTH_SHORT).show()
        }
    }

    private fun stopVoiceCommand() {
        try {
            speechRecognizer?.stopListening()
            isListening = false
            btnVoiceCommand.text = "语音指令"
            Log.d("IntegratedNavi", "🎤 停止语音识别")
        } catch (e: Exception) {
            Log.e("IntegratedNavi", "❌ 停止语音识别失败: ${e.message}")
        }
    }



    // ==================== 语音识别回调 ====================
    override fun onReadyForSpeech(params: Bundle?) {
        Log.d("IntegratedNavi", "🎤 语音识别准备就绪")
    }

    override fun onBeginningOfSpeech() {
        Log.d("IntegratedNavi", "🎤 开始说话")
    }

    override fun onRmsChanged(rmsdB: Float) {
        // 音量变化，可用于UI反馈
    }

    override fun onBufferReceived(buffer: ByteArray?) {
        // 缓冲区数据接收
    }

    override fun onEndOfSpeech() {
        Log.d("IntegratedNavi", "🎤 说话结束")
        isListening = false
        btnVoiceCommand.text = "语音指令"
    }

    override fun onError(error: Int) {
        val errorMsg = when (error) {
            SpeechRecognizer.ERROR_AUDIO -> "音频错误，请检查麦克风"
            SpeechRecognizer.ERROR_CLIENT -> "客户端错误"
            SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "录音权限不足，请在设置中授权"
            SpeechRecognizer.ERROR_NETWORK -> "网络错误，请检查网络连接"
            SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "网络超时，请重试"
            SpeechRecognizer.ERROR_NO_MATCH -> "无法识别，请清晰说出目的地"
            SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "识别器繁忙，请稍候重试"
            SpeechRecognizer.ERROR_SERVER -> "服务器错误，请稍候重试"
            SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "说话超时，请重试"
            else -> "未知错误($error)，请重试"
        }

        Log.e("IntegratedNavi", "❌ 语音识别错误: $errorMsg ($error)")

        isListening = false
        btnVoiceCommand.text = "语音指令"

        // 修复问题一：ERROR 后销毁识别器，下次 doStartListening 重新创建干净实例
        speechRecognizer?.destroy()
        speechRecognizer = null

        val voiceMsg = when (error) {
            SpeechRecognizer.ERROR_NO_MATCH -> "未识别到语音，请清晰说出目的地"
            SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "说话时间太短，请重新说"
            SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "需要录音权限，请在设置中授权"
            SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "识别器繁忙，请稍候"
            else -> "语音识别失败，请重试"
        }

        addVoiceMessage(VoiceMessage(MessagePriority.INFO, voiceMsg, MessageType.SYSTEM))
        Toast.makeText(this, errorMsg, Toast.LENGTH_SHORT).show()
    }

    override fun onResults(results: Bundle?) {
        val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
        val bestMatch = matches?.firstOrNull()

        if (!bestMatch.isNullOrEmpty()) {
            Log.d("IntegratedNavi", "🎤 识别结果: $bestMatch")
            handleVoiceCommand(bestMatch)
        } else {
            addVoiceMessage(VoiceMessage(MessagePriority.INFO, "未识别到语音，请重试", MessageType.SYSTEM))
        }
    }

    override fun onPartialResults(partialResults: Bundle?) {
        // 部分结果，可用于实时反馈
    }

    override fun onEvent(eventType: Int, params: Bundle?) {
        // 事件回调
    }

    // ==================== 语音指令处理 ====================
    private fun handleVoiceCommand(command: String) {
        val cleanCommand = command.trim()
        Log.d("IntegratedNavi", "🎤 处理指令：$cleanCommand, waitingForDestination=$waitingForDestination")

        // 如果正在等待确认
        if (waitingForDestination) {
            Log.d("IntegratedNavi", "🔍 等待确认阶段，用户输入: $cleanCommand")

            // 扩展确认关键词，提高识别率
            val confirmKeywords = listOf("确认", "是的", "对", "好", "开始", "出发", "可以", "行", "没问题", "确定")
            val cancelKeywords = listOf("取消", "不对", "错误", "不", "重新", "不要", "错了", "换一个")

            val isConfirm = confirmKeywords.any { cleanCommand.contains(it) }
            val isCancel = cancelKeywords.any { cleanCommand.contains(it) }

            if (isConfirm) {
                Log.d("IntegratedNavi", "✅ 用户确认目的地: $tempDestination")
                waitingForDestination = false
                addVoiceMessage(VoiceMessage(MessagePriority.IMPORTANT, "确认前往${tempDestination}，开始导航", MessageType.SYSTEM))
                startNavigation(tempDestination)
            } else if (isCancel) {
                Log.d("IntegratedNavi", "❌ 用户取消目的地: $tempDestination")
                waitingForDestination = false
                tempDestination = ""
                addVoiceMessage(VoiceMessage(MessagePriority.INFO, "好的，请重新告诉我要去哪里", MessageType.SYSTEM))
                voiceHandler.postDelayed({ startVoiceCommand() }, 2000)
            } else {
                Log.d("IntegratedNavi", "❓ 未识别确认指令，用户输入: $cleanCommand")
                // 用户说了其他内容，再次询问
                addVoiceMessage(VoiceMessage(MessagePriority.INFO, "请问您是要去${tempDestination}吗？请说确认或取消", MessageType.SYSTEM))
                voiceHandler.postDelayed({ startVoiceCommand() }, 2000)
            }
            return
        }

        // 正常处理语音指令
        when {
            cleanCommand.contains("去") || cleanCommand.contains("导航") || cleanCommand.contains("怎么走") -> {
                val dest = cleanCommand.replace(Regex("(帮我|我要|我想|带我|导航到|去|怎么走)"), "").trim()
                if (dest.isNotEmpty()) {
                    // 询问用户确认
                    tempDestination = dest
                    waitingForDestination = true
                    addVoiceMessage(VoiceMessage(MessagePriority.INFO, "请问您是要去${dest}吗？请说确认或取消", MessageType.SYSTEM))
                    // 2 秒后自动开始语音识别，让用户可以直接回答
                    voiceHandler.postDelayed({
                        if (waitingForDestination) {
                            startVoiceCommand()
                        }
                    }, 2000)
                } else {
                    addVoiceMessage(VoiceMessage(MessagePriority.INFO, "请告诉我目的地", MessageType.SYSTEM))
                    voiceHandler.postDelayed({ startVoiceCommand() }, 2000)
                }
            }
            cleanCommand.contains("取消") || cleanCommand.contains("停止") -> {
                stopNavigation()
                addVoiceMessage(VoiceMessage(MessagePriority.INFO, "导航已取消", MessageType.SYSTEM))
            }
            cleanCommand.contains("帮助") || cleanCommand.contains("怎么用") || cleanCommand.contains("说明") -> {
                addVoiceMessage(VoiceMessage(MessagePriority.INFO, "可以说：去武汉大学、去光谷广场、去汉口火车站等地点，或说确认开始导航，取消停止导航", MessageType.SYSTEM))
            }
            else -> {
                addVoiceMessage(VoiceMessage(MessagePriority.INFO, "没有理解，可以说：去某地，或者取消导航", MessageType.SYSTEM))
                voiceHandler.postDelayed({ startVoiceCommand() }, 2500)
            }
        }
    }



    // ==================== 导航功能 ====================
    private fun startNavigation(dest: String) {
        if (currentLocation == null) {
            addVoiceMessage(VoiceMessage(MessagePriority.INFO, "正在获取当前位置，请稍候", MessageType.SYSTEM))
            return
        }

        waitingForDestination = false

        // 地理编码
        geocodeAddress(dest) { latLng ->
            if (latLng != null) {
                destinationLatLng = latLng
                calculateRoute(latLng)
            } else {
                addVoiceMessage(VoiceMessage(MessagePriority.IMPORTANT, "无法找到目的地位置", MessageType.SYSTEM))
            }
        }
    }

    private fun stopNavigation() {
        try {
            aMapNavi.stopNavi()
            updateInstruction("导航已停止")
            updateNext("等待指令")
            updateDistance("-")
            updateTime("-")
        } catch (e: Exception) {
            Log.e("IntegratedNavi", "❌ 停止导航失败: ${e.message}")
        }
    }

    private fun calculateRoute(destination: NaviLatLng) {
        try {
            val startPoint = currentLocation ?: NaviLatLng(30.5438, 114.3638) // 默认武汉大学
            val result = aMapNavi.calculateWalkRoute(startPoint, destination)
            Log.d("IntegratedNavi", "步行路线规划请求结果: $result")

            if (result) {
                addVoiceMessage(VoiceMessage(MessagePriority.INFO, "正在规划路线", MessageType.NAVIGATION))
            } else {
                addVoiceMessage(VoiceMessage(MessagePriority.IMPORTANT, "路线规划失败", MessageType.SYSTEM))
            }
        } catch (e: Exception) {
            Log.e("IntegratedNavi", "❌ 路线规划失败: ${e.message}")
            addVoiceMessage(VoiceMessage(MessagePriority.IMPORTANT, "路线规划失败", MessageType.SYSTEM))
        }
    }

    // ==================== 地理编码 ====================
    private fun geocodeAddress(address: String, callback: (NaviLatLng?) -> Unit) {
        try {
            if (geocodeSearch == null) {
                geocodeSearch = GeocodeSearch(this)
                geocodeSearch?.setOnGeocodeSearchListener(this)
            }

            pendingGeocodeCallback = callback
            val query = GeocodeQuery(address, "")
            geocodeSearch?.getFromLocationNameAsyn(query)

        } catch (e: Exception) {
            Log.e("IntegratedNavi", "❌ 地理编码失败: ${e.message}")
            callback(null)
        }
    }

    override fun onGeocodeSearched(result: GeocodeResult?, rCode: Int) {
        if (rCode == AMapException.CODE_AMAP_SUCCESS && result != null && result.geocodeAddressList.isNotEmpty()) {
            val latLonPoint = result.geocodeAddressList[0].latLonPoint
            val naviLatLng = NaviLatLng(latLonPoint.latitude, latLonPoint.longitude)
            Log.d("IntegratedNavi", "✅ 地理编码成功: (${naviLatLng.latitude}, ${naviLatLng.longitude})")
            pendingGeocodeCallback?.invoke(naviLatLng)
        } else {
            Log.e("IntegratedNavi", "❌ 地理编码失败，错误码: $rCode")
            pendingGeocodeCallback?.invoke(null)
        }
        pendingGeocodeCallback = null
    }

    override fun onRegeocodeSearched(result: RegeocodeResult?, rCode: Int) {
        // 逆地理编码暂不需要
    }

    // ==================== 导航信息更新 ====================
    // 修复问题二：记录上次播报的导航指令，避免每个定位 tick 都入队同一条消息
    // 原代码每次 onNaviInfoUpdate 都 addVoiceMessage(IMPORTANT)，
    // 高优先级消息持续占满队列，障碍消息永远排不上队
    private var lastNavInstruction = ""
    private var lastNavSpeakTime = 0L
    private val navSpeakIntervalMs = 8000L // 导航语音最快每8秒一次

    private fun updateNavigationInfo(info: NaviInfo) {
        val distance = info.curStepRetainDistance
        val time = info.curStepRetainTime
        val instruction = "前方 ${distance} 米，约 ${time / 60} 分钟"

        updateInstruction(instruction)
        updateDistance("剩余 ${distance} 米")
        updateTime("约 ${time / 60} 分钟")

        val now = System.currentTimeMillis()
        // 只在指令变化 或 超过间隔时间 才入队播报，防止刷队列
        if (instruction != lastNavInstruction && now - lastNavSpeakTime > navSpeakIntervalMs) {
            addVoiceMessage(VoiceMessage(MessagePriority.IMPORTANT, instruction, MessageType.NAVIGATION))
            lastNavInstruction = instruction
            lastNavSpeakTime = now
        }
    }

    // ==================== UI更新方法 ====================
    private fun updateInstruction(text: String) {
        runOnUiThread { tvInstruction.text = text }
    }

    private fun updateDistance(text: String) {
        runOnUiThread { tvDistance.text = "距离: $text" }
    }

    private fun updateTime(text: String) {
        runOnUiThread { tvTime.text = "时间: $text" }
    }

    private fun updateNext(text: String) {
        runOnUiThread { tvNext.text = text }
    }

    private fun updateDestination(text: String) {
        runOnUiThread { tvDestination.text = "目的地: $text" }
    }

    // ==================== TTS相关 ====================
    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts.setLanguage(Locale.SIMPLIFIED_CHINESE)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("IntegratedNavi", "❌ TTS不支持中文")
            } else {
                ttsReady = true
                tts.setSpeechRate(0.9f)
                tts.setPitch(1.0f)

                // 修复问题一：用真实 TTS 播报完成回调驱动队列，
                // 替代原来的延迟估算，消除时序错位导致语音识别无响应的问题
                tts.setOnUtteranceProgressListener(object : android.speech.tts.UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {}
                    override fun onDone(utteranceId: String?) {
                        voiceHandler.post {
                            isSpeaking = false
                            currentSpeakingMessage = null
                            processVoiceQueue()
                        }
                    }
                    @Deprecated("Deprecated in Java")
                    override fun onError(utteranceId: String?) {
                        voiceHandler.post {
                            isSpeaking = false
                            currentSpeakingMessage = null
                            processVoiceQueue()
                        }
                    }
                })

                Log.d("IntegratedNavi", "✅ TTS初始化成功")
                addVoiceMessage(VoiceMessage(MessagePriority.INFO, "集成导航系统已启动", MessageType.SYSTEM))
            }
        } else {
            Log.e("IntegratedNavi", "❌ TTS初始化失败")
        }
    }

    // 修复问题一：speak() 必须带 utteranceId，UtteranceProgressListener 才会回调
    private fun speak(text: String) {
        if (ttsReady) {
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "utt_${System.currentTimeMillis()}")
        }
    }

    // ==================== 生命周期管理 ====================
    override fun onDestroy() {
        super.onDestroy()

        // 修复：先停止监听再销毁，避免 destroy 时内部状态异常
        try { speechRecognizer?.stopListening() } catch (_: Exception) {}
        speechRecognizer?.destroy()
        speechRecognizer = null

        tts.stop()
        tts.shutdown()

        locationListener?.let { listener ->
            locationManager.removeUpdates(listener)
        }

        try {
            aMapNavi.stopNavi()
        } catch (e: Exception) {
            Log.e("IntegratedNavi", "❌ 导航资源清理失败: ${e.message}")
        }

        cameraProvider?.unbindAll()

        voiceHandler.removeCallbacksAndMessages(null)
        detectionHandler.removeCallbacksAndMessages(null)

        Log.d("IntegratedNavi", "✅ 集成导航页面已销毁")
    }
}