package com.example.myapplication

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import android.view.animation.LinearInterpolator
import kotlin.math.sin

class WaveformView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#4CAF50")
        style = Paint.Style.FILL
    }

    // 柱子数量
    private val barCount = 24

    // 动态计算的柱子宽度（根据屏幕宽度自动调整）
    private var barWidth = 0f

    // 柱子间距
    private val barSpacing = 6f

    private var amplitudes = FloatArray(barCount) { 0.3f }
    private var animator: ValueAnimator? = null
    private var phase = 0f

    init {
        startAnimation()
    }

    fun startAnimation() {
        if (animator?.isRunning == true) return

        animator = ValueAnimator.ofFloat(0f, 360f).apply {
            duration = 800
            repeatCount = ValueAnimator.INFINITE
            interpolator = LinearInterpolator()
            addUpdateListener {
                phase = it.animatedValue as Float
                updateAmplitudes()
                invalidate()
            }
            start()
        }
    }

    fun stopAnimation() {
        animator?.cancel()
        amplitudes.fill(0.2f)
        invalidate()
    }

    private fun updateAmplitudes() {
        for (i in amplitudes.indices) {
            val angle = phase + i * 15f
            amplitudes[i] = (0.3f + 0.5f * sin(Math.toRadians(angle.toDouble())).toFloat())
                .coerceIn(0.2f, 1f)
        }
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        // 计算柱子宽度：占屏幕宽度的80%，减去间距后除以柱子数量
        val targetWidth = (w * 0.8f).toInt()  // 占屏幕宽度的80%
        val totalSpacing = (barCount - 1) * barSpacing
        barWidth = (targetWidth - totalSpacing) / barCount.toFloat()
        barWidth = barWidth.coerceIn(6f, 20f)  // 限制最小6dp，最大20dp
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()
        val centerY = viewHeight / 2

        // 计算总宽度（占屏幕宽度的80%）
        val totalWidth = barCount * barWidth + (barCount - 1) * barSpacing
        // 居中显示
        val startX = (viewWidth - totalWidth) / 2f

        for (i in amplitudes.indices) {
            val barHeight = viewHeight * amplitudes[i] * 0.6f
            val left = startX + i * (barWidth + barSpacing)
            val top = centerY - barHeight / 2
            val right = left + barWidth
            val bottom = centerY + barHeight / 2

            canvas.drawRoundRect(left, top, right, bottom, barWidth / 2, barWidth / 2, paint)
        }
    }
}