import { useState, useEffect } from 'react'
import { getMetrics } from '../api/client'
import './ModelMetricsChart.css'

const METRIC_KEYS = [
  { key: 'accuracy', label: 'Accuracy', format: (v) => `${(v * 100).toFixed(2)}%`, max: 1 },
  { key: 'balanced_accuracy', label: 'Balanced accuracy', format: (v) => `${(v * 100).toFixed(2)}%`, max: 1 },
  { key: 'f1_rest', label: 'F1 Rest', format: (v) => (v ?? 0).toFixed(4), max: 1 },
  { key: 'f1_left', label: 'F1 Left Hand', format: (v) => (v ?? 0).toFixed(4), max: 1 },
  { key: 'f1_right', label: 'F1 Right Hand', format: (v) => (v ?? 0).toFixed(4), max: 1 },
  { key: 'macro_f1', label: 'Macro F1', format: (v) => (v ?? 0).toFixed(4), max: 1 },
  { key: 'latency_ms', label: 'Latency (ms)', format: (v) => (v ?? 0).toFixed(2), max: null },
]

export default function ModelMetricsChart() {
  const [metrics, setMetrics] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    getMetrics()
      .then((data) => { if (!cancelled) setMetrics(Array.isArray(data) ? data : []) })
      .catch((e) => { if (!cancelled) setError(e.message) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [])

  if (loading) return <div className="metrics-chart-panel"><p className="metrics-loading">Loading metrics…</p></div>
  if (error) return <div className="metrics-chart-panel"><p className="metrics-error">{error}</p></div>
  if (metrics.length === 0) {
    return (
      <div className="metrics-chart-panel">
        <h2 className="panel-title">Model comparison: accuracy &amp; latency</h2>
        <p className="metrics-empty">No metrics yet. Run <code>python3 train_model.py</code> to generate eval_results.json.</p>
      </div>
    )
  }

  const maxLatency = Math.max(...metrics.map((m) => m.latency_ms || 0), 1)

  return (
    <div className="metrics-chart-panel">
      <h2 className="panel-title">Model comparison: accuracy &amp; latency</h2>
      <div className="metrics-table-wrap">
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Model</th>
              {METRIC_KEYS.map(({ label }) => (
                <th key={label}>{label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metrics.map((row, i) => (
              <tr key={row.model ?? i}>
                <td className="model-name">{row.model}</td>
                {METRIC_KEYS.map(({ key, format }) => (
                  <td key={key}>{format(row[key])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="metrics-bars">
        {METRIC_KEYS.filter((m) => m.max !== null && m.key !== 'latency_ms').map(({ key, label, max }) => (
          <div key={key} className="metric-bar-row">
            <span className="metric-bar-label">{label}</span>
            <div className="metric-bar-track">
              {metrics.map((row, i) => (
                <div key={i} className="metric-bar-segment" style={{ flex: 1 }}>
                  <div
                    className="metric-bar"
                    style={{
                      width: `${Math.min(100, ((row[key] ?? 0) / max) * 100)}%`,
                      backgroundColor: `hsl(${220 - i * 40}, 60%, 50%)`,
                    }}
                    title={`${row.model}: ${(row[key] ?? 0).toFixed(2)}`}
                  />
                </div>
              ))}
            </div>
          </div>
        ))}
        <div className="metric-bar-row">
          <span className="metric-bar-label">Latency (ms) — lower is better</span>
          <div className="metric-bar-track">
            {metrics.map((row, i) => (
              <div key={i} className="metric-bar-segment" style={{ flex: 1 }}>
                <div
                  className="metric-bar metric-bar--latency"
                  style={{
                    width: `${Math.min(100, ((row.latency_ms ?? 0) / maxLatency) * 100)}%`,
                    backgroundColor: `hsl(${280 - i * 40}, 60%, 50%)`,
                  }}
                  title={`${row.model}: ${(row.latency_ms ?? 0).toFixed(2)} ms`}
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
