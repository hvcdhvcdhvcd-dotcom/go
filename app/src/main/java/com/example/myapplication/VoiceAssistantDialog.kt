package com.example.myapplication

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.view.WindowManager
import android.view.animation.AlphaAnimation
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentDialog
import androidx.activity.OnBackPressedCallback

class VoiceAssistantDialog(context: Context) : ComponentDialog(context, android.R.style.Theme_Black_NoTitleBar_Fullscreen) {

    private lateinit var waveformView: WaveformView
    private lateinit var micIcon: ImageView
    lateinit var tvRecognizedText: TextView
    lateinit var tvHint: TextView
    private lateinit var btnCancel: Button
    private lateinit var btnConfirm: Button

    private var pulseAnimator: ValueAnimator? = null
    private var onConfirmListener: ((String) -> Unit)? = null
    private var onCancelListener: (() -> Unit)? = null

    var recognizedText: String = ""
        set(value) {
            field = value
            tvRecognizedText.text = if (value.isNotEmpty()) value else "请说话..."
        }

    companion object {
        fun show(
            context: Context,
            onConfirm: (String) -> Unit,
            onCancel: (() -> Unit)? = null
        ): VoiceAssistantDialog {
            val dialog = VoiceAssistantDialog(context)
            dialog.onConfirmListener = onConfirm
            dialog.onCancelListener = onCancel
            dialog.show()
            return dialog
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
        setupAnimations()
        setupBackCallback()

        startWaveformAnimation()
    }

    private fun initViews() {
        waveformView = findViewById(R.id.waveformView)
        micIcon = findViewById(R.id.micIcon)
        tvRecognizedText = findViewById(R.id.tvRecognizedText)
        tvHint = findViewById(R.id.tvHint)
        btnCancel = findViewById(R.id.btnCancel)
        btnConfirm = findViewById(R.id.btnConfirm)

        btnCancel.setOnClickListener {
            onCancelListener?.invoke()
            dismiss()
        }

        btnConfirm.setOnClickListener {
            if (recognizedText.isNotEmpty()) {
                onConfirmListener?.invoke(recognizedText)
                dismiss()
            } else {
                tvHint.text = "请先说出目的地"
                tvHint.setTextColor(Color.parseColor("#FF5722"))
                tvHint.postDelayed({
                    tvHint.text = "说出目的地，如：去武汉大学"
                    tvHint.setTextColor(Color.parseColor("#AAAAAA"))
                }, 2000)
            }
        }
    }

    // ========== UI 动画方法（供外部调用） ==========

    fun startListeningAnimation() {
        startWaveformAnimation()
        startMicPulseAnimation()
        tvHint.text = "正在聆听，请说话..."
        tvHint.setTextColor(Color.parseColor("#4CAF50"))
        tvRecognizedText.text = "正在聆听..."
    }

    fun stopListeningAnimation() {
        stopWaveformAnimation()
        stopMicPulseAnimation()
        tvHint.text = "说出目的地，如：去武汉大学"
        tvHint.setTextColor(Color.parseColor("#AAAAAA"))
    }

    fun onRecognitionError(errorMsg: String) {
        stopWaveformAnimation()
        stopMicPulseAnimation()
        tvHint.text = errorMsg
        tvHint.setTextColor(Color.parseColor("#FF5722"))
        tvRecognizedText.text = errorMsg
        tvHint.postDelayed({
            if (isShowing) {
                tvHint.text = "说出目的地，如：去武汉大学"
                tvHint.setTextColor(Color.parseColor("#AAAAAA"))
                tvRecognizedText.text = "请说话..."
            }
        }, 2000)
    }

    fun updatePartialText(text: String) {
        tvRecognizedText.text = text
    }

    // ========== 私有动画方法 ==========

    private fun startWaveformAnimation() {
        waveformView.startAnimation()
    }

    private fun stopWaveformAnimation() {
        waveformView.stopAnimation()
    }

    private fun startMicPulseAnimation() {
        stopMicPulseAnimation()
        pulseAnimator = ValueAnimator.ofArgb(
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
    }

    private fun stopMicPulseAnimation() {
        pulseAnimator?.cancel()
        micIcon.background?.setTint(Color.parseColor("#1976D2"))
    }

    private fun setupAnimations() {
        val fadeIn = AlphaAnimation(0f, 1f).apply {
            duration = 300
        }
        findViewById<View>(R.id.waveformView)?.startAnimation(fadeIn)
    }

    private fun setupBackCallback() {
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                onCancelListener?.invoke()
                dismiss()
            }
        })
    }

    override fun dismiss() {
        stopWaveformAnimation()
        stopMicPulseAnimation()
        super.dismiss()
    }
}