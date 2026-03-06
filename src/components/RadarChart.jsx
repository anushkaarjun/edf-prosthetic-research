import { useRef, useEffect } from 'react'
import './RadarChart.css'

export default function RadarChart({ labels, values, size = 200 }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !labels?.length || !values?.length) return

    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    canvas.width = size * dpr
    canvas.height = size * dpr
    ctx.scale(dpr, dpr)

    const cx = size / 2
    const cy = size / 2
    const radius = (size / 2) - 28
    const n = Math.min(labels.length, values.length)
    const maxVal = Math.max(...values, 1)

    // Grid circles
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

    // Axes
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

    // Data polygon
    const normalized = values.slice(0, n).map(v => Math.min(1, Math.max(0, v / maxVal)))
    ctx.fillStyle = 'rgba(34, 211, 238, 0.25)'
    ctx.strokeStyle = '#22d3ee'
    ctx.lineWidth = 2
    ctx.beginPath()
    for (let i = 0; i < n; i++) {
      const angle = (i / n) * 2 * Math.PI - Math.PI / 2
      const r = radius * (0.1 + 0.9 * normalized[i])
      const x = cx + r * Math.cos(angle)
      const y = cy + r * Math.sin(angle)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.closePath()
    ctx.fill()
    ctx.stroke()

    // Labels
    ctx.fillStyle = '#a1a1aa'
    ctx.font = '10px monospace'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    for (let i = 0; i < n; i++) {
      const angle = (i / n) * 2 * Math.PI - Math.PI / 2
      const x = cx + (radius + 14) * Math.cos(angle)
      const y = cy + (radius + 14) * Math.sin(angle)
      ctx.fillText(labels[i], x, y)
    }
  }, [labels, values, size])

  return (
    <div className="radar-wrap">
      <canvas ref={canvasRef} className="radar-canvas" width={size} height={size} />
      <p className="radar-title">Channel power (radar)</p>
    </div>
  )
}
