/**
 * Connect to your model backend.
 * Set VITE_MODEL_URL in .env (e.g. VITE_MODEL_URL=http://localhost:5000/api/predict)
 * Your API can accept POST with { channels: number, sampleRate: number, samples: number[][] }
 * and return e.g. { predictions: number[], radarValues?: number[] } or whatever your model uses.
 */

const MODEL_URL = import.meta.env.VITE_MODEL_URL || ''

export async function fetchModelPrediction(payload) {
  if (!MODEL_URL) return null
  try {
    const res = await fetch(MODEL_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!res.ok) throw new Error(`Model API ${res.status}`)
    return res.json()
  } catch (err) {
    console.warn('Model request failed:', err.message)
    return null
  }
}

export function isModelConfigured() {
  return Boolean(MODEL_URL)
}
