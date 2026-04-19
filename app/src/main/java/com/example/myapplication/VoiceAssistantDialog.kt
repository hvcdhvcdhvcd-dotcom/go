package com.example.myapplication

import android.animation.ValueAnimator
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.view.View
import android.view.WindowManager
import android.view.animation.AlphaAnimation
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentDialog
import androidx.activity.OnBackPressedCallback
import androidx.core.graphics.drawable.toDrawable

class VoiceAssistantDialog(context: Context) : ComponentDialog(context, android.R.style.Theme_Black_NoTitleBar_Fullscreen) {

    private lateinit var waveformView: WaveformView
    private lateinit var micIcon: ImageView
    private lateinit var tvRecognizedText: TextView
    private lateinit var tvHint: TextView
    private lateinit var btnCancel: Button
    private lateinit var btnConfirm: Button

    private var speechRecognizer: SpeechRecognizer? = null
    private var isListening = false
    private var recognizedText = ""
    private var onDestinationConfirmed: ((String) -> Unit)? = null

    companion object {
        fun show(context: Context, onConfirm: (String) -> Unit) {
            val dialog = VoiceAssistantDialog(context)
            dialog.onDestinationConfirmed = onConfirm
            dialog.show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.dialog_voice_assistant)

        window?.setLayout(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.MATCH_PARENT
        )
        window?.setBackgroundDrawableResource(android.R.color.transparent)

        initViews()
        initSpeechRecognizer()
        startListening()
        setupAnimations()

        // ✅ 修复：使用 AndroidX 的 OnBackPressedDispatcher 处理返回键
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                stopListening()
                dismiss()
            }
        })
    }

    private fun initViews() {
        waveformView = findViewById(R.id.waveformView)
        micIcon = findViewById(R.id.micIcon)
        tvRecognizedText = findViewById(R.id.tvRecognizedText)
        tvHint = findViewById(R.id.tvHint)
        btnCancel = findViewById(R.id.btnCancel)
        btnConfirm = findViewById(R.id.btnConfirm)

        btnCancel.setOnClickListener {
            stopListening()
            dismiss()
        }

        btnConfirm.setOnClickListener {
            if (recognizedText.isNotEmpty()) {
                onDestinationConfirmed?.invoke(recognizedText)
                stopListening()
                dismiss()
            } else {
                tvHint.text = "请先说出目的地"
                tvHint.setTextColor(android.graphics.Color.parseColor("#FF5722"))
                tvHint.postDelayed({
                    tvHint.text = "说出目的地，如：去武汉大学"
                    tvHint.setTextColor(android.graphics.Color.parseColor("#AAAAAA"))
                }, 2000)
            }
        }
    }

    private fun initSpeechRecognizer() {
        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            tvHint.text = "设备不支持语音识别"
            tvHint.setTextColor(android.graphics.Color.parseColor("#FF5722"))
            return
        }

        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context)
        speechRecognizer?.setRecognitionListener(object : android.speech.RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                isListening = true
                waveformView.startAnimation()
                startMicPulseAnimation()
                tvRecognizedText.text = "正在聆听..."
                tvHint.text = "请说话..."
            }

            override fun onBeginningOfSpeech() {
                tvHint.text = "正在识别..."
            }

            override fun onRmsChanged(rmsdB: Float) {}

            override fun onBufferReceived(buffer: ByteArray?) {}

            override fun onEndOfSpeech() {
                tvHint.text = "识别完成"
            }

            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                val text = matches?.firstOrNull() ?: ""

                if (text.isNotEmpty()) {
                    recognizedText = text
                    tvRecognizedText.text = text
                    tvHint.text = "确认导航或重新说话"

                    val destination = extractDestination(text)
                    if (destination.isNotEmpty()) {
                        recognizedText = destination
                        tvRecognizedText.text = "去 $destination"
                    }

                    stopWaveformAnimation()
                    stopMicPulseAnimation()
                } else {
                    tvRecognizedText.text = "未识别到语音"
                    tvHint.text = "请重新说话"
                    startListening()
                }
                isListening = false
            }

            override fun onError(error: Int) {
                val errorMsg = when (error) {
                    SpeechRecognizer.ERROR_NO_MATCH -> "未匹配到语音"
                    SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "说话超时"
                    SpeechRecognizer.ERROR_AUDIO -> "录音错误"
                    SpeechRecognizer.ERROR_NETWORK -> "网络错误"
                    else -> "识别失败"
                }
                tvRecognizedText.text = errorMsg
                tvHint.text = "请重试"
                stopWaveformAnimation()
                stopMicPulseAnimation()
                isListening = false

                tvHint.postDelayed({
                    if (isShowing) startListening()
                }, 2000)
            }

            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
    }

    private fun startListening() {
        if (speechRecognizer == null) return

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "zh-CN")
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            putExtra(RecognizerIntent.EXTRA_PROMPT, "说出目的地")
        }
        speechRecognizer?.startListening(intent)
        isListening = true
    }

    private fun stopListening() {
        speechRecognizer?.stopListening()
        isListening = false
        stopWaveformAnimation()
        stopMicPulseAnimation()
    }

    private fun startMicPulseAnimation() {
        val pulseAnim = ValueAnimator.ofArgb(
            Color.parseColor("#1976D2"),
            Color.parseColor("#4CAF50"),
            Color.parseColor("#1976D2")
        ).apply {
            duration = 1000
            repeatCount = ValueAnimator.INFINITE
            addUpdateListener { animator ->
                val color = animator.animatedValue as Int
                micIcon.background?.setTint(color)
            }
            start()
        }
        micIcon.tag = pulseAnim
    }

    private fun stopMicPulseAnimation() {
        (micIcon.tag as? ValueAnimator)?.cancel()
        micIcon.background?.setTint(Color.parseColor("#1976D2"))
    }

    private fun stopWaveformAnimation() {
        waveformView.stopAnimation()
    }

    private fun setupAnimations() {
        val fadeIn = AlphaAnimation(0f, 1f).apply {
            duration = 300
        }
        findViewById<View>(R.id.waveformView)?.startAnimation(fadeIn)
    }

    private fun extractDestination(text: String): String {
        val patterns = listOf("去", "导航到", "我要去", "想去", "到")
        for (pattern in patterns) {
            if (text.contains(pattern)) {
                val index = text.indexOf(pattern) + pattern.length
                val dest = text.substring(index).trim()
                if (dest.isNotEmpty()) {
                    return dest
                }
            }
        }
        return text
    }

    override fun dismiss() {
        stopListening()
        speechRecognizer?.destroy()
        super.dismiss()
    }
}