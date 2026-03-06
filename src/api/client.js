import { API_BASE } from '../constants'

export async function getStatus() {
  const res = await fetch(`${API_BASE}/api/status`)
  if (!res.ok) throw new Error('API not reachable')
  return res.json()
}

export async function getValidationSample() {
  const res = await fetch(`${API_BASE}/api/validation/sample?t=${Date.now()}`, {
    cache: 'no-store',
    headers: { Pragma: 'no-cache', 'Cache-Control': 'no-cache' },
  })
  if (!res.ok) throw new Error('Failed to get validation sample')
  return res.json()
}

export async function predict(eegSegment) {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ eeg: eegSegment }),
    cache: 'no-store',
  })
  if (!res.ok) throw new Error('Predict failed')
  return res.json()
}

export async function getMetrics() {
  const res = await fetch(`${API_BASE}/api/metrics`)
  if (!res.ok) return []
  return res.json()
}
