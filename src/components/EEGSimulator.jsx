import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend } from 'recharts';
import { Check } from 'lucide-react';
import { getStatus, getValidationSample } from '../api/client';
import { SAMPLE_RATE } from '../constants';

// 3-class: Left Hand, Right Hand, Rest — same colors for EEG traces and radar
const MOTOR_CLASSES = [
  { name: 'Left Hand', color: '#3b82f6', bgColor: 'rgba(59, 130, 246, 0.1)' },
  { name: 'Right Hand', color: '#ef4444', bgColor: 'rgba(239, 68, 68, 0.1)' },
  { name: 'Rest', color: '#6b7280', bgColor: 'rgba(107, 114, 136, 0.1)' }
];
const EEG_TRACE_COLORS = [MOTOR_CLASSES[0].color, MOTOR_CLASSES[1].color, MOTOR_CLASSES[2].color, MOTOR_CLASSES[0].color]; // ch1=Left, ch2=Right, ch3=Rest, ch4=Left

const CLASS_NAMES = MOTOR_CLASSES.map(c => c.name);

function eegToChartData(eeg) {
  if (!eeg || !eeg.length) return [];
  const nTime = Array.isArray(eeg[0]) ? eeg[0].length : 0;
  if (nTime === 0) return [];
  const fs = SAMPLE_RATE;
  const data = [];
  for (let i = 0; i < nTime; i++) {
    const t = i / fs;
    data.push({
      time: t.toFixed(2),
      ch1: Number(eeg[0]?.[i] ?? 0),
      ch2: Number(eeg[1]?.[i] ?? 0),
      ch3: Number(eeg[2]?.[i] ?? 0),
      ch4: Number(eeg[3]?.[i] ?? 0)
    });
  }
  return data;
}

function getClassIndex(name) {
  const i = CLASS_NAMES.indexOf(name);
  return i >= 0 ? i : 0;
}

const EEGSimulator = () => {
  const [apiConnected, setApiConnected] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelName, setModelName] = useState('');
  const [error, setError] = useState(null);
  const [playing, setPlaying] = useState(false);

  const [eegData, setEegData] = useState([]);
  const [probabilities, setProbabilities] = useState(MOTOR_CLASSES.map(() => 1 / MOTOR_CLASSES.length));
  const [actualClass, setActualClass] = useState(0);
  const [predictedClass, setPredictedClass] = useState(0);
  const [isCorrect, setIsCorrect] = useState(false);
  const [loadingSample, setLoadingSample] = useState(false);

  const loadNextSample = useCallback(async () => {
    if (!apiConnected) return;
    setError(null);
    setLoadingSample(true);
    try {
      const sample = await getValidationSample();
      const eeg = sample.eeg ?? sample.data;
      if (!eeg || !Array.isArray(eeg) || eeg.length === 0) {
        setError('No EEG data in response. Check backend logs.');
        return;
      }
      const actualName = sample.actualClass ?? CLASS_NAMES[sample.actualClassIndex ?? 0];
      const actualIdx = getClassIndex(actualName);

      setEegData(eegToChartData(eeg));
      setActualClass(actualIdx);

      // Use probabilities from validation sample (computed on backend, no round-trip)
      const probs = sample.probabilities ?? sample.probs;
      let probsArr = [];
      if (Array.isArray(probs)) {
        probsArr = probs.slice(0, MOTOR_CLASSES.length).map(p => Number(p));
      } else if (probs && typeof probs === 'object') {
        probsArr = CLASS_NAMES.map(name => {
          let v = probs[name] ?? 0;
          if (typeof v !== 'number' || Number.isNaN(v)) v = 0;
          if (v > 1) v = v / 100;
          return v;
        });
      }
      if (probsArr.length < MOTOR_CLASSES.length) {
        probsArr = [...probsArr, ...Array(MOTOR_CLASSES.length - probsArr.length).fill(0)];
      }
      const sum = probsArr.reduce((a, b) => a + b, 0);
      const normalized = sum > 0 ? probsArr.map(p => p / sum) : MOTOR_CLASSES.map(() => 1 / MOTOR_CLASSES.length);
      setProbabilities(normalized);

      const predName = sample.predictedClassName ?? (sample.classes && sample.classes[sample.predictedClass]) ?? CLASS_NAMES[sample.predictedClass] ?? CLASS_NAMES[0];
      const predIdx = getClassIndex(predName);
      setPredictedClass(predIdx);
      setIsCorrect(predIdx === actualIdx);
    } catch (e) {
      setError(e.message || 'Failed to load sample');
    } finally {
      setLoadingSample(false);
    }
  }, [apiConnected]);

  const fetchStatus = useCallback(async () => {
    try {
      const s = await getStatus();
      setApiConnected(true);
      setModelLoaded(s.modelLoaded ?? false);
      setModelName(s.modelName ?? '');
      setError(null);
    } catch {
      setApiConnected(false);
      setModelLoaded(false);
      setModelName('');
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const t = setInterval(fetchStatus, 5000);
    return () => clearInterval(t);
  }, [fetchStatus]);

  useEffect(() => {
    if (!playing || !apiConnected) return;
    loadNextSample();
    const id = setInterval(loadNextSample, 3000);
    return () => clearInterval(id);
  }, [playing, apiConnected, loadNextSample]);

  // Radar shows model prediction: 3 classes (Left Hand, Right Hand, Rest) with confidence %
  const radarData = MOTOR_CLASSES.map((cls, idx) => ({
    class: cls.name,
    confidence: Number(((probabilities[idx] ?? 0) * 100).toFixed(2)),
    fill: cls.color
  }));

  const bgColor = MOTOR_CLASSES[predictedClass].bgColor;
  const predictedColor = MOTOR_CLASSES[predictedClass].color;

  return (
    <div className="w-full min-h-screen p-6 relative" style={{ backgroundColor: bgColor, transition: 'background-color 0.5s' }}>
      {/* Status bar */}
      <div className={`mb-4 px-4 py-2 rounded-lg ${apiConnected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
        API: {apiConnected ? 'Connected' : 'Disconnected'}
        {apiConnected && (modelLoaded ? ` · Visualization: ${modelName || '3-class CNN-LSTM'}` : ' · No Model')}
      </div>
      {!apiConnected && (
        <div className="mb-4 p-4 rounded-lg bg-amber-50 border border-amber-200 text-amber-900">
          <p className="font-semibold mb-2">Start the backend to use the app:</p>
          <p className="text-sm mb-2">In a terminal run:</p>
          <pre className="bg-white p-3 rounded text-sm overflow-x-auto mb-2">cd backend && python3 app.py</pre>
          <p className="text-sm mb-2">Wait for &quot;[API] Starting on http://0.0.0.0:5001&quot;, then click Retry below.</p>
          <button
            type="button"
            onClick={fetchStatus}
            className="px-4 py-2 rounded-lg font-semibold bg-amber-500 text-white hover:bg-amber-600"
          >
            Retry connection
          </button>
        </div>
      )}
      {error && (
        <div className="mb-4 px-4 py-2 rounded-lg bg-red-200 text-red-900">
          {error}
        </div>
      )}

      {/* Correct Prediction Indicator */}
      {isCorrect && (
        <div className="absolute top-20 left-1/2 transform -translate-x-1/2 z-10">
          <div className="bg-green-500 text-white rounded-full p-3 shadow-lg animate-pulse">
            <Check size={32} strokeWidth={3} />
          </div>
        </div>
      )}

      <div className="grid grid-cols-3 gap-6">
        {/* Left: EEG Signals — colors by class (Left Hand, Right Hand, Rest) */}
        <div className="col-span-1 bg-white rounded-lg shadow-lg p-4 min-h-[400px]">
          <h2 className="text-lg font-bold mb-4 text-gray-800">EEG Signals (4 Channels)</h2>
          <p className="text-xs text-gray-500 mb-2">Ch1: Left Hand · Ch2: Right Hand · Ch3: Rest · Ch4: Left Hand</p>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={eegData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="time" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Line type="monotone" dataKey="ch1" stroke={EEG_TRACE_COLORS[0]} dot={false} strokeWidth={1.5} name="Ch1" />
              <Line type="monotone" dataKey="ch2" stroke={EEG_TRACE_COLORS[1]} dot={false} strokeWidth={1.5} name="Ch2" />
              <Line type="monotone" dataKey="ch3" stroke={EEG_TRACE_COLORS[2]} dot={false} strokeWidth={1.5} name="Ch3" />
              <Line type="monotone" dataKey="ch4" stroke={EEG_TRACE_COLORS[3]} dot={false} strokeWidth={1.5} name="Ch4" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Center: Probability Bars */}
        <div className="col-span-1 bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 text-center">Motor Function Probability</h2>
          <p className="text-xs text-gray-500 mb-4 text-center">Confidence varies by sample; values are model outputs, not fixed.</p>
          <div className="space-y-6">
            {MOTOR_CLASSES.map((cls, idx) => (
              <div key={idx}>
                <div className="flex justify-between mb-2">
                  <span className="font-semibold text-gray-700">{cls.name}</span>
                  <span className="font-bold" style={{ color: cls.color }}>
                    {((probabilities[idx] ?? 0) * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-8 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-300 flex items-center justify-end pr-2"
                    style={{
                      width: `${((probabilities[idx] ?? 0) * 100).toFixed(2)}%`,
                      backgroundColor: cls.color
                    }}
                  >
                    {(probabilities[idx] ?? 0) > 0.15 && (
                      <span className="text-white text-xs font-bold">
                        {((probabilities[idx] ?? 0) * 100).toFixed(1)}%
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-8 p-4 bg-gray-50 rounded-lg">
            <div className="flex justify-between mb-2">
              <span className="font-semibold">Actual Class:</span>
              <span style={{ color: MOTOR_CLASSES[actualClass].color }} className="font-bold">
                {MOTOR_CLASSES[actualClass].name}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="font-semibold">Predicted Class:</span>
              <span style={{ color: MOTOR_CLASSES[predictedClass].color }} className="font-bold">
                {MOTOR_CLASSES[predictedClass].name}
              </span>
            </div>
          </div>
        </div>

        {/* Right: Radar Chart — model prediction for Left Hand, Right Hand, Rest; colored by predicted class */}
        <div className="col-span-1 bg-white rounded-lg shadow-lg p-6 min-h-[400px]">
          <h2 className="text-xl font-bold mb-4 text-gray-800 text-center">Confidence Radar</h2>
          <p className="text-xs text-gray-500 mb-2 text-center">Model confidence for each class · Highlight: {MOTOR_CLASSES[predictedClass].name}</p>
          <p className="text-xs text-gray-400 mb-2 text-center italic">Confidence varies by sample (e.g. 99.8% vs 100%).</p>
          <ResponsiveContainer width="100%" height={350}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#e5e7eb" />
              <PolarAngleAxis
                dataKey="class"
                tick={({ payload, x, y }) => (
                  <text x={x} y={y} textAnchor="middle" fontSize={12} fill={payload.value === MOTOR_CLASSES[predictedClass].name ? predictedColor : '#374151'} fontWeight={payload.value === MOTOR_CLASSES[predictedClass].name ? 700 : 400}>
                    {payload.value}
                  </text>
                )}
              />
              <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 10 }} />
              <Radar
                name="Confidence %"
                dataKey="confidence"
                stroke={predictedColor}
                fill={predictedColor}
                fillOpacity={0.5}
                strokeWidth={2}
              />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Controls */}
      {loadingSample && (
        <div className="mb-4 px-4 py-2 rounded-lg bg-blue-100 text-blue-800 text-center">
          Loading sample… (first one can take 10–30 s)
        </div>
      )}
      <div className="mt-6 flex gap-4 justify-center">
        <button
          type="button"
          onClick={() => setPlaying(p => !p)}
          disabled={!apiConnected || loadingSample}
          className="px-6 py-3 rounded-lg font-semibold bg-blue-600 text-white disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-700"
        >
          {playing ? 'Stop' : 'Run validation'}
        </button>
        <button
          type="button"
          onClick={loadNextSample}
          disabled={!apiConnected || loadingSample}
          className="px-6 py-3 rounded-lg font-semibold bg-gray-600 text-white disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-700"
        >
          Next sample
        </button>
      </div>
    </div>
  );
};

export default EEGSimulator;
