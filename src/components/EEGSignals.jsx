import { useRef, useEffect } from 'react'
import { CHANNELS_DISPLAYED, SAMPLE_RATE } from '../constants'
import './EEGSignals.css'

// Trace colors by class: Rest (blue), Left Hand (green), Right Hand (orange)
const COLORS_REST = ['#22d3ee', '#38bdf8', '#0ea5e9', '#0284c7']
const COLORS_LEFT_HAND = ['#34d399', '#22c55e', '#16a34a', '#15803d']
const COLORS_RIGHT_HAND = ['#fb923c', '#f97316', '#ea580c', '#c2410c']

// data: original input EEG [64][time]. Colored by predicted class.
export default function EEGSignals({ data, sampleRate = SAMPLE_RATE, predictedClass, probabilities }) {
  const canvasRef = useRef(null)
  const fs = Number(sampleRate) || SAMPLE_RATE

  const theme = predictedClass === 'Rest'
    ? 'rest'
    : predictedClass === 'Left Hand'
      ? 'left-hand'
      : 'right-hand'
  const traceColors = theme === 'rest' ? COLORS_REST : theme === 'left-hand' ? COLORS_LEFT_HAND : COLORS_RIGHT_HAND

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !data?.length) return

    const nCh = Math.min(CHANNELS_DISPLAYED, data.length)
    const series0 = Array.isArray(data[0]) ? data[0] : []
    const nTime = series0.length
    if (nCh === 0 || nTime === 0) return

    const durationSec = nTime / fs

    let minVal = Infinity
    let maxVal = -Infinity
    for (let ch = 0; ch < nCh; ch++) {
      const series = Array.isArray(data[ch]) ? data[ch] : []
      for (let i = 0; i < series.length; i++) {
        const v = Number(series[i])
        if (!Number.isNaN(v)) {
          minVal = Math.min(minVal, v)
          maxVal = Math.max(maxVal, v)
        }
      }
    }
    const range = maxVal - minVal || 1
    const padding = range * 0.1 || 1
    const dataMin = minVal - padding
    const dataMax = maxVal + padding
    const dataRange = dataMax - dataMin

    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)
    const w = rect.width
    const h = rect.height

    const marginLeft = 44
    const marginRight = 12
    const marginTop = 12
    const marginBottom = 28
    const plotW = w - marginLeft - marginRight
    const plotH = h - marginTop - marginBottom
    const channelHeight = plotH / nCh
    const stepX = plotW / Math.max(1, nTime - 1)
    const yMid = (i) => marginTop + (i + 0.5) * channelHeight
    const plotHeight = channelHeight * 0.82

    // Background tint by prediction
    const tint = theme === 'rest' ? 'rgba(14, 165, 233, 0.06)' : theme === 'left-hand' ? 'rgba(34, 197, 94, 0.06)' : 'rgba(249, 115, 22, 0.06)'
    ctx.fillStyle = tint
    ctx.fillRect(marginLeft, marginTop, plotW, plotH)
    ctx.fillStyle = '#0f0f12'
    ctx.fillRect(0, 0, marginLeft, h)
    ctx.fillRect(marginLeft + plotW, 0, w - marginLeft - plotW, h)

    // Left border: strong color for current prediction
    const borderColor = theme === 'rest' ? '#0ea5e9' : theme === 'left-hand' ? '#22c55e' : '#f97316'
    ctx.fillStyle = borderColor
    ctx.fillRect(0, 0, 4, h)

    // Grid
    ctx.strokeStyle = '#27272a'
    ctx.lineWidth = 0.5
    for (let ch = 0; ch <= nCh; ch++) {
      ctx.beginPath()
      ctx.moveTo(marginLeft, marginTop + ch * channelHeight)
      ctx.lineTo(w - marginRight, marginTop + ch * channelHeight)
      ctx.stroke()
    }
    for (let t = 0; t <= durationSec; t += 1) {
      const x = marginLeft + (t / durationSec) * plotW
      ctx.beginPath()
      ctx.moveTo(x, marginTop)
      ctx.lineTo(x, h - marginBottom)
      ctx.stroke()
    }

    // Traces (original EEG) – color by Rest vs Right Hand prediction
    for (let ch = 0; ch < nCh; ch++) {
      const series = Array.isArray(data[ch]) ? data[ch] : []
      if (!series.length) continue

      ctx.strokeStyle = traceColors[ch % traceColors.length]
      ctx.lineWidth = 1.5
      ctx.beginPath()
      for (let i = 0; i < series.length; i++) {
        const x = marginLeft + i * stepX
        const v = Number(series[i])
        const norm = Number.isNaN(v) ? 0.5 : (v - dataMin) / dataRange
        const y = yMid(ch) - (norm - 0.5) * plotHeight
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.stroke()
    }

    // Y-axis
    ctx.fillStyle = '#71717a'
    ctx.font = '10px monospace'
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    const yTicks = [dataMax, (dataMin + dataMax) / 2, dataMin]
    yTicks.forEach((val, i) => {
      const y = marginTop + (i / 2) * plotH
      ctx.fillText(`${Number(val).toFixed(1)} µV`, marginLeft - 6, y)
    })
    ctx.save()
    ctx.translate(14, marginTop + plotH / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.fillText('Amplitude (µV)', 0, 0)
    ctx.restore()

    // X-axis
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    const nTicks = 6
    const tickStep = durationSec / (nTicks - 1) || 1
    for (let i = 0; i < nTicks; i++) {
      const t = i * tickStep
      const x = marginLeft + (t / durationSec) * plotW
      ctx.fillText(Number(t).toFixed(2), x, h - marginBottom + 4)
    }
    ctx.fillText('Time (s)', marginLeft + plotW / 2, h - 6)

    // Channel labels
    ctx.textAlign = 'left'
    ctx.fillStyle = '#a1a1aa'
    ctx.font = '10px monospace'
    for (let ch = 0; ch < nCh; ch++) {
      const y = yMid(ch)
      ctx.fillText(`Ch ${ch + 1}`, marginLeft + plotW + 4, y)
    }
  }, [data, fs, theme, traceColors])

  return (
    <div className={`eeg-panel eeg-panel--${theme}`}>
      <h2 className="panel-title">EEG signals (4 of 64 channels)</h2>
      <p className="eeg-meta">
        {data?.length
          ? `${SAMPLE_RATE} Hz · ${(Array.isArray(data[0]) ? data[0].length : 0) / SAMPLE_RATE} s · `
          : '—'}
        <span className={`eeg-legend eeg-legend--rest ${theme === 'rest' ? 'eeg-legend--active' : ''}`}>■ Rest</span>
        <span className={`eeg-legend eeg-legend--left ${theme === 'left-hand' ? 'eeg-legend--active' : ''}`}> ■ Left hand</span>
        <span className={`eeg-legend eeg-legend--right ${theme === 'right-hand' ? 'eeg-legend--active' : ''}`}> ■ Right hand</span>
      </p>
      <div className="eeg-canvas-wrap">
        <canvas ref={canvasRef} className="eeg-canvas" />
      </div>
    </div>
  )
}
