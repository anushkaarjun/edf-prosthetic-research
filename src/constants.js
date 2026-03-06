// 3-class: Rest, Left Hand, Right Hand (motor imagery)
export const MOTOR_CLASSES = ['Rest', 'Left Hand', 'Right Hand']

export const TOTAL_CHANNELS = 64
export const CHANNELS_DISPLAYED = 4
export const SAMPLE_RATE = 160 // common for EEG motor imagery (e.g. BCI IV 2a)
// In dev, Vite proxies /api to backend (localhost:5001). Use '' so requests go to same origin.
export const API_BASE = import.meta.env.VITE_API_URL ?? ''
