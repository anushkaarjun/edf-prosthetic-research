import { MOTOR_CLASSES } from '../constants'
import './MotorProbability.css'

export default function MotorProbability({ probabilities, actualClass, predictedClass }) {
  const probs = probabilities || {}
  const list = MOTOR_CLASSES.map((label, i) => {
    let v = probs[label] ?? probs[i] ?? 0
    if (v > 1) v = v / 100
    return { label, value: v }
  })

  return (
    <div className="motor-panel">
      <h2 className="panel-title motor-title">
        Motor Function Probability
        <span className="check-icon" title="Ready">✓</span>
      </h2>
      <p className="motor-meta">P(Rest), P(Left hand), P(Right hand) from CNN-LSTM</p>
      <ul className="motor-list">
        {list.map(({ label, value }) => (
          <li key={label} className="motor-row">
            <span className="motor-label">{label}</span>
            <span className="motor-pct">{Number((value * 100).toFixed(1))}%</span>
            <div className="motor-bar-wrap">
              <div className="motor-bar" style={{ width: `${Math.min(100, value * 100)}%` }} />
            </div>
          </li>
        ))}
      </ul>
      <div className="class-result">
        <div><strong>Actual Class:</strong> {actualClass ?? '—'}</div>
        <div><strong>Predicted Class:</strong> {predictedClass ?? '—'}</div>
      </div>
    </div>
  )
}
