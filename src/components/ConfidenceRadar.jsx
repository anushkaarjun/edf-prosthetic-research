import { useRef, useEffect } from 'react'
import { MOTOR_CLASSES } from '../constants'
import './ConfidenceRadar.css'

export default function ConfidenceRadar({ values }) {
  const canvasRef = useRef(null)
  const labels = MOTOR_CLASSES
  const vals = values ?? labels.map(() => 0)
  const size = 220

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    canvas.width = size * dpr
    canvas.height = size * dpr
    ctx.scale(dpr, dpr)

    const cx = size / 2
    const cy = size / 2
    const radius = (size / 2) - 36
    const n = Math.min(labels.length, vals.length)
    const maxVal = Math.max(...vals, 1)

    ctx.strokeStyle = '#27272a'
    ctx.lineWidth = 0.5
    for (let r = 0.25; r <= 1; r += 0.25) {
      ctx.beginPath()
      for (let i = 0; i <= n; i++) {
        const angle = (i / n) * 2 * Math.PI - Math.PI / 2
        const x = cx + radius * r * Math.cos(angle)
        const y = cy + radius * r * Math.sin(angle)
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.closePath()
      ctx.stroke()
    }

    for (let i = 0; i < n; i++) {
      const angle = (i / n) * 2 * Math.PI - Math.PI / 2
      const x = cx + radius * Math.cos(angle)
      const y = cy + radius * Math.sin(angle)
      ctx.strokeStyle = '#3f3f46'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(cx, cy)
      ctx.lineTo(x, y)
      ctx.stroke()
    }

    const normalized = vals.slice(0, n).map(v => Math.min(1, Math.max(0, Number(v) / maxVal)))
    ctx.fillStyle = 'rgba(168, 85, 247, 0.35)'
    ctx.strokeStyle = '#a855f7'
    ctx.lineWidth = 2
    ctx.beginPath()
    for (let i = 0; i < n; i++) {
      const angle = (i / n) * 2 * Math.PI - Math.PI / 2
      const r = radius * (0.05 + 0.95 * normalized[i])
      const x = cx + r * Math.cos(angle)
      const y = cy + r * Math.sin(angle)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.closePath()
    ctx.fill()
    ctx.stroke()

    ctx.fillStyle = '#a1a1aa'
    ctx.font = '9px system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    for (let i = 0; i < n; i++) {
      const angle = (i / n) * 2 * Math.PI - Math.PI / 2
      const x = cx + (radius + 20) * Math.cos(angle)
      const y = cy + (radius + 20) * Math.sin(angle)
      ctx.fillText(labels[i], x, y)
    }
  }, [values])

  return (
    <div className="confidence-radar-panel">
      <h2 className="panel-title">Confidence Radar</h2>
      <canvas ref={canvasRef} className="confidence-radar-canvas" width={size} height={size} />
      <div className="radar-legend">
        <span className="radar-legend-swatch" />
        <span>Confidence %</span>
      </div>
    </div>
  )
}
