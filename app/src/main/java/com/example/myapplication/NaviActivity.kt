package com.example.myapplication

import android.app.Activity
import android.Manifest
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.camera.core.ImageProxy
import androidx.core.app.ActivityCompat
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
import java.util.Locale

class NaviActivity : Activity(), TextToSpeech.OnInitListener, GeocodeSearch.OnGeocodeSearchListener {

    // ==================== 导航相关变量 ====================
    private lateinit var aMapNavi: AMapNavi
    private var destination: String = ""

    // ==================== UI 组件 ====================
    private lateinit var tvInstruction: TextView
    private lateinit var tvDistance: TextView
    private lateinit var tvTime: TextView
    private lateinit var tvNext: TextView
    private lateinit var btnExit: Button

    // ==================== 定位相关 ====================
    private lateinit var locationManager: LocationManager
    private var currentLocation: Location? = null

    // ==================== 语音播报 ====================
    private lateinit var tts: TextToSpeech
    private var isTtsReady = false

    // ==================== 导航状态 ====================
    private var isNavigating = false

    // ==================== 障碍检测（暂未实现）====================
    private val mainHandler = Handler(Looper.getMainLooper())

    // ==================== 地理编码 ====================
    private lateinit var geocodeSearch: GeocodeSearch
    private var pendingGeocodeCallback: ((NaviLatLng?) -> Unit)? = null

    private var startPoint: NaviLatLng? = null
    private var endPoint: NaviLatLng? = null

    // ==================== 导航监听器（使用 SimpleNaviListener 适配器）====================
    private val naviListener = object : SimpleNaviListener() {

        override fun onGetNavigationText(type: Int, text: String?) {
            text?.let {
                updateInstruction(it)
                speak(it)
            }
        }

        override fun onCalculateRouteSuccess(routeResult: AMapCalcRouteResult) {
            Log.d("NaviActivity", "步行路线规划成功")
            updateInstruction("路线规划成功，开始导航")
            speak("路线规划成功，开始导航")
            aMapNavi.startNavi(NaviType.GPS)
        }

        override fun onCalculateRouteFailure(routeResult: AMapCalcRouteResult) {
            Log.e("NaviActivity", "步行路线规划失败，错误详情：${routeResult.errorDetail}")
            updateInstruction("路线规划失败，请检查网络或地址")
            speak("路线规划失败")
        }

        override fun onInitNaviFailure() {
            Log.e("NaviActivity", "导航初始化失败")
            speak("导航初始化失败")
        }

        override fun onStartNavi(type: Int) {
            Log.d("NaviActivity", "导航已开始")
            isNavigating = true
            updateInstruction("导航已开始，请直行")
            speak("导航已开始，请直行")
        }

        override fun onNaviInfoUpdate(naviInfo: NaviInfo?) {
            naviInfo?.let {
                // 获取剩余距离（米）和剩余时间（秒）
                val distance = it.curStepRetainDistance
                val time = it.curStepRetainTime
                updateDistance("剩余 ${distance} 米")
                updateTime("约 ${time / 60} 分钟")
            }
        }

        override fun onArriveDestination() {
            speak("已到达目的地")
            updateInstruction("已到达目的地")
            stopNavigation()
            finish()
        }

        override fun onLocationChange(location: AMapNaviLocation?) {
            // 可选：处理实时位置更新
        }
    }

    // ==================== Activity 生命周期 ====================
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 高德地图隐私合规
        try {
            MapsInitializer.updatePrivacyShow(this, true, true)
            MapsInitializer.updatePrivacyAgree(this, true)
            Log.d("NaviActivity", "隐私合规设置成功")
        } catch (e: Exception) {
            Log.e("NaviActivity", "隐私合规设置失败：${e.message}", e)
        }

        setContentView(R.layout.activity_navi)

        initViews()

        destination = intent.getStringExtra("dest") ?: ""
        if (destination.isEmpty()) {
            Toast.makeText(this, "目的地不能为空", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        tts = TextToSpeech(this, this)
        locationManager = getSystemService(LOCATION_SERVICE) as LocationManager
        initNavi()
    }

    override fun onDestroy() {
        super.onDestroy()
        stopNavigation()
        tts.shutdown()
    }

    override fun onResume() {
        super.onResume()
    }

    override fun onPause() {
        super.onPause()
    }

    // ==================== UI 初始化 ====================
    private fun initViews() {
        tvInstruction = findViewById(R.id.navi_instruction)
        tvDistance = findViewById(R.id.navi_distance)
        tvTime = findViewById(R.id.navi_time)
        tvNext = findViewById(R.id.navi_next)
        btnExit = findViewById(R.id.btn_exit_navi)

        btnExit.setOnClickListener {
            stopNavigation()
            finish()
        }
    }

    // ==================== 高德导航初始化 ====================
    private fun initNavi() {
        try {
            aMapNavi = AMapNavi.getInstance(applicationContext)
            aMapNavi.addAMapNaviListener(naviListener)
            startNavigationFlow()
        } catch (e: Exception) {
            Log.e("NaviActivity", "初始化导航失败：${e.message}")
            Toast.makeText(this, "初始化导航失败", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    // ==================== 完整导航流程 ====================
    private fun startNavigationFlow() {
        speak("导航系统已启动")

        getCurrentLocation { location ->
            currentLocation = location
            Log.d("NaviActivity", "获取到当前位置：${location?.latitude}, ${location?.longitude}")
            updateInstruction("位置获取成功，正在查询目的地...")
            speak("位置获取成功，正在查询" + destination + "的位置")

            geocodeAddress(destination) { targetLatLng ->
                targetLatLng?.let { end ->
                    Log.d("NaviActivity", "目的地坐标：${end.latitude}, ${end.longitude}")
                    updateInstruction("目的地已找到，正在规划路线...")
                    speak("目的地已找到，正在规划前往" + destination + "的路线")
                    calculateWalkingRoute(location!!, end)
                } ?: run {
                    updateInstruction("未找到目的地，请重试")
                    speak("未找到" + destination + "的位置，请重试")
                }
            }
        }
    }

    // ==================== 获取当前位置 ====================
    private fun getCurrentLocation(callback: (Location?) -> Unit) {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED
        ) {
            Toast.makeText(this, "需要定位权限", Toast.LENGTH_SHORT).show()
            callback(null)
            return
        }

        updateInstruction("正在获取您的位置...")

        val provider = LocationManager.GPS_PROVIDER
        try {
            currentLocation = locationManager.getLastKnownLocation(provider)
            if (currentLocation != null) {
                callback(currentLocation)
            } else {
                Log.d("NaviActivity", "未获取到位置，等待定位...")
                locationManager.requestSingleUpdate(provider, object : LocationListener {
                    override fun onLocationChanged(location: Location) {
                        currentLocation = location
                        callback(location)
                    }
                    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
                    override fun onProviderEnabled(provider: String) {}
                    override fun onProviderDisabled(provider: String) {}
                }, Looper.getMainLooper())

                mainHandler.postDelayed({
                    if (currentLocation == null) {
                        Log.d("NaviActivity", "定位超时，使用默认位置")
                        currentLocation = Location(provider).apply {
                            latitude = 30.5430
                            longitude = 114.3560
                        }
                        callback(currentLocation)
                    }
                }, 5000)
            }
        } catch (e: Exception) {
            Log.e("NaviActivity", "定位失败：${e.message}")
            currentLocation = Location(provider).apply {
                latitude = 30.5430
                longitude = 114.3560
            }
            callback(currentLocation)
        }
    }

    // ==================== 地理编码 ====================
    private fun geocodeAddress(address: String, callback: (NaviLatLng?) -> Unit) {
        speak("正在查询" + address + "的位置")
        updateInstruction("正在查询目的地位置...")
        Log.d("NaviActivity", "🔍 地理编码查询：$address")

        geocodeSearch = GeocodeSearch(this)
        geocodeSearch.setOnGeocodeSearchListener(this)
        val query = GeocodeQuery(address, "武汉")
        geocodeSearch.getFromLocationNameAsyn(query)

        pendingGeocodeCallback = callback
    }

    // ==================== 计算步行路线 ====================
    private fun calculateWalkingRoute(start: Location, end: NaviLatLng) {
        try {
            startPoint = NaviLatLng(start.latitude, start.longitude)
            endPoint = end

            Log.d("NaviActivity", "起点：${startPoint?.latitude}, ${startPoint?.longitude}")
            Log.d("NaviActivity", "终点：${endPoint?.latitude}, ${endPoint?.longitude}")

            updateInstruction("正在规划步行路线...")
            speak("正在规划步行路线")

            val result = aMapNavi.calculateWalkRoute(startPoint, endPoint)
            Log.d("NaviActivity", "步行路线规划请求结果：$result")
            if (!result) {
                updateInstruction("路线规划失败，请重试")
                speak("路线规划失败")
            }
        } catch (e: Exception) {
            Log.e("NaviActivity", "路线计算失败：${e.message}", e)
            updateInstruction("路线规划失败")
            speak("路线规划失败")
        }
    }

    // ==================== UI 更新 ====================
    private fun updateInstruction(text: String) {
        runOnUiThread { tvInstruction.text = text }
    }

    private fun updateDistance(text: String) {
        runOnUiThread { tvDistance.text = text }
    }

    private fun updateTime(text: String) {
        runOnUiThread { tvTime.text = text }
    }

    @Suppress("unused")
    private fun updateNext(text: String) {
        runOnUiThread { tvNext.text = text }
    }

    // ==================== 语音播报 ====================
    private fun speak(text: String) {
        if (isTtsReady) {
            tts.speak(text, TextToSpeech.QUEUE_ADD, null, null)
        }
    }

    // ==================== 停止导航 ====================
    private fun stopNavigation() {
        try {
            if (::aMapNavi.isInitialized) {
                aMapNavi.stopNavi()
            }
            if (::geocodeSearch.isInitialized) {
                geocodeSearch.setOnGeocodeSearchListener(null)
            }
            if (isTtsReady) {
                tts.stop()
            }
            Log.d("NaviActivity", "导航已停止")
        } catch (e: Exception) {
            Log.e("NaviActivity", "停止导航失败：${e.message}")
        }
    }

    // ==================== TTS 初始化回调 ====================
    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts.setLanguage(Locale.CHINESE)
            if (result != TextToSpeech.LANG_MISSING_DATA && result != TextToSpeech.LANG_NOT_SUPPORTED) {
                isTtsReady = true
                tts.setSpeechRate(1.0f)
                Log.d("NaviActivity", "TTS 初始化成功")
            } else {
                Log.e("NaviActivity", "TTS 语言设置失败")
            }
        } else {
            Log.e("NaviActivity", "TTS 初始化失败")
        }
    }

    // ==================== 地理编码回调 ====================
    override fun onGeocodeSearched(result: GeocodeResult?, rCode: Int) {
        if (rCode == AMapException.CODE_AMAP_SUCCESS && result != null && result.geocodeAddressList.isNotEmpty()) {
            val latLonPoint = result.geocodeAddressList[0].latLonPoint
            val naviLatLng = NaviLatLng(latLonPoint.latitude, latLonPoint.longitude)
            Log.d("NaviActivity", "✅ 地理编码成功：(${naviLatLng.latitude}, ${naviLatLng.longitude})")
            pendingGeocodeCallback?.invoke(naviLatLng)
        } else {
            Log.e("NaviActivity", "❌ 地理编码失败，错误码：$rCode")
            pendingGeocodeCallback?.invoke(null)
        }
        pendingGeocodeCallback = null
    }

    override fun onRegeocodeSearched(result: RegeocodeResult?, rCode: Int) {
        // 逆地理编码暂不需要
    }
}

// ==================== 辅助数据类（障碍检测用，暂未实现完整功能）====================
data class DetectionResult(
    val label: String,
    val confidence: Float,
    val bbox: android.graphics.RectF
)

class ObjectDetector(
    private val context: android.content.Context,
    private val modelPath: String
) {
    fun detect(imageProxy: ImageProxy): List<DetectionResult> {
        return emptyList()
    }
}