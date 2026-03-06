import './PredictionHistory.css'

const MAX_ENTRIES = 16

export default function PredictionHistory({ history = [] }) {
  const list = history.slice(-MAX_ENTRIES).reverse()

  return (
    <div className="prediction-history-panel">
      <h2 className="panel-title">Prediction history</h2>
      <p className="history-meta">Last {MAX_ENTRIES} samples · ✓ = correct</p>
      <div className="history-table-wrap">
        <table className="history-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Actual</th>
              <th>Predicted</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {list.length === 0 ? (
              <tr><td colSpan={4} className="history-empty">Run validation or click Next sample</td></tr>
            ) : (
              list.map((entry, i) => {
                const correct = entry.actualClass === entry.predictedClass
                return (
                  <tr key={i} className={correct ? 'correct' : 'incorrect'}>
                    <td>{list.length - i}</td>
                    <td>{entry.actualClass ?? '—'}</td>
                    <td>{entry.predictedClass ?? '—'}</td>
                    <td>{correct ? '✓' : '✗'}</td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
      {list.length > 0 && (
        <p className="history-accuracy">
          Accuracy (this page): {list.filter(e => e.actualClass === e.predictedClass).length} / {list.length} = {Math.round(100 * list.filter(e => e.actualClass === e.predictedClass).length / list.length)}%
        </p>
      )}
    </div>
  )
}
