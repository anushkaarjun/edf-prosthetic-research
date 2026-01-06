import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend } from 'recharts';
import { Check } from 'lucide-react';

// Updated motor classes to match Python model (Both Fists instead of Tongue)
const MOTOR_CLASSES = [
  { name: 'Left Hand', color: '#3b82f6', bgColor: 'rgba(59, 130, 246, 0.1)' },
  { name: 'Right Hand', color: '#ef4444', bgColor: 'rgba(239, 68, 68, 0.1)' },
  { name: 'Both Feet', color: '#10b981', bgColor: 'rgba(16, 185, 129, 0.1)' },
  { name: 'Both Fists', color: '#f59e0b', bgColor: 'rgba(245, 158, 11, 0.1)' },
  { name: 'Rest', color: '#6b7280', bgColor: 'rgba(107, 114, 136, 0.1)' }
];

const EEGSimulator = () => {
  const [eegData, setEegData] = useState([]);
  const [probabilities, setProbabilities] = useState(MOTOR_CLASSES.map(() => 0.2));
  const [actualClass, setActualClass] = useState(0);
  const [predictedClass, setPredictedClass] = useState(0);
  const [isCorrect, setIsCorrect] = useState(false);
  const [useAPI, setUseAPI] = useState(false);
  const [apiStatus, setApiStatus] = useState('disconnected');
  const timeRef = useRef(0);
  const maxDataPoints = 100;
  const API_URL = 'http://localhost:8000';

  // Generate realistic EEG signal matching Python preprocessing
  const generateEEGSignal = (t, freq, amp, noise) => {
    // Multi-frequency signal similar to real EEG (8-30 Hz band)
    const base = amp * Math.sin(2 * Math.PI * freq * t);
    const alpha = 0.3 * amp * Math.sin(2 * Math.PI * (freq * 0.8) * t);
    const beta = 0.2 * amp * Math.sin(2 * Math.PI * (freq * 1.2) * t);
    const noise_component = (Math.random() - 0.5) * noise;
    return base + alpha + beta + noise_component;
  };

  // Check API health
  useEffect(() => {
    const checkAPI = async () => {
      try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        if (data.model_loaded) {
          setApiStatus('connected');
          setUseAPI(true);
        } else {
          setApiStatus('no_model');
        }
      } catch (error) {
        setApiStatus('disconnected');
        setUseAPI(false);
      }
    };
    checkAPI();
    const interval = setInterval(checkAPI, 5000);
    return () => clearInterval(interval);
  }, []);

  // Fetch prediction from API
  const fetchPrediction = async (channels) => {
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          channels: channels,
          sample_rate: 250.0
        })
      });
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('API prediction error:', error);
      return null;
    }
  };

  useEffect(() => {
    const interval = setInterval(async () => {
      timeRef.current += 0.05;
      const t = timeRef.current;

      // Generate multi-channel EEG data (simulating 4 representative channels)
      // In real scenario, this would be all 64 channels
      const channels = [];
      const displayedChannels = [];
      
      // Generate 64 channels for API (if using), but display only 4
      for (let ch = 0; ch < 64; ch++) {
        const freq = 8 + (ch % 10) * 2.2; // Vary frequency per channel (8-30 Hz range)
        const amp = 40 + Math.random() * 20;
        const signal = generateEEGSignal(t, freq, amp, 15);
        channels.push([signal]);
        
        // Store first 4 for display
        if (ch < 4) {
          displayedChannels.push(signal);
        }
      }

      // For display, use 4 channels
      const newPoint = {
        time: t.toFixed(2),
        ch1: displayedChannels[0] || generateEEGSignal(t, 10, 50, 20),
        ch2: displayedChannels[1] || generateEEGSignal(t, 12, 45, 18),
        ch3: displayedChannels[2] || generateEEGSignal(t, 8, 55, 22),
        ch4: displayedChannels[3] || generateEEGSignal(t, 15, 40, 15)
      };

      setEegData(prev => {
        const updated = [...prev, newPoint];
        return updated.slice(-maxDataPoints);
      });

      // Get predictions
      let newProbs = MOTOR_CLASSES.map(() => 0.2);
      
      if (useAPI && apiStatus === 'connected') {
        // Use real API prediction
        const prediction = await fetchPrediction(channels);
        if (prediction && prediction.probabilities) {
          newProbs = prediction.probabilities;
        }
      } else {
        // Simulated probabilities with smooth transitions
        newProbs = MOTOR_CLASSES.map((_, idx) => {
          const base = Math.random() * 0.3;
          const boost = idx === actualClass ? 0.5 + Math.random() * 0.3 : 0;
          return Math.min(base + boost, 1);
        });
        
        // Normalize probabilities
        const sum = newProbs.reduce((a, b) => a + b, 0);
        newProbs = newProbs.map(p => p / sum);
      }

      setProbabilities(newProbs);

      // Determine predicted class
      const maxProb = Math.max(...newProbs);
      const predicted = newProbs.indexOf(maxProb);
      setPredictedClass(predicted);
      setIsCorrect(predicted === actualClass);

      // Randomly change actual class every 5 seconds (simulated)
      if (Math.random() < 0.01) {
        setActualClass(Math.floor(Math.random() * MOTOR_CLASSES.length));
      }
    }, 50);

    return () => clearInterval(interval);
  }, [actualClass, useAPI, apiStatus]);

  // Prepare radar chart data
  const radarData = MOTOR_CLASSES.map((cls, idx) => ({
    class: cls.name,
    confidence: (probabilities[idx] * 100).toFixed(1)
  }));

  const bgColor = MOTOR_CLASSES[predictedClass].bgColor;

  // API status indicator
  const getApiStatusColor = () => {
    switch (apiStatus) {
      case 'connected': return 'bg-green-500';
      case 'no_model': return 'bg-yellow-500';
      default: return 'bg-red-500';
    }
  };

  const getApiStatusText = () => {
    switch (apiStatus) {
      case 'connected': return 'API Connected (Using Real Model)';
      case 'no_model': return 'API Connected (No Model Loaded)';
      default: return 'API Disconnected (Using Simulation)';
    }
  };

  return (
    <div className="w-full h-screen p-6 relative" style={{ backgroundColor: bgColor, transition: 'background-color 0.5s' }}>
      {/* API Status Indicator */}
      <div className="absolute top-4 right-4 z-10">
        <div className={`${getApiStatusColor()} text-white rounded-lg px-4 py-2 shadow-lg text-sm font-semibold`}>
          {getApiStatusText()}
        </div>
      </div>

      {/* Correct Prediction Indicator */}
      {isCorrect && (
        <div className="absolute top-8 left-1/2 transform -translate-x-1/2 z-10">
          <div className="bg-green-500 text-white rounded-full p-3 shadow-lg animate-pulse">
            <Check size={32} strokeWidth={3} />
          </div>
        </div>
      )}

      <div className="grid grid-cols-3 gap-6 h-full">
        {/* Left: EEG Signals */}
        <div className="col-span-1 bg-white rounded-lg shadow-lg p-4">
          <h2 className="text-lg font-bold mb-4 text-gray-800">EEG Signals (4 of 64 Channels)</h2>
          <ResponsiveContainer width="100%" height="90%">
            <LineChart data={eegData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="time" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Line type="monotone" dataKey="ch1" stroke="#3b82f6" dot={false} strokeWidth={1.5} />
              <Line type="monotone" dataKey="ch2" stroke="#ef4444" dot={false} strokeWidth={1.5} />
              <Line type="monotone" dataKey="ch3" stroke="#10b981" dot={false} strokeWidth={1.5} />
              <Line type="monotone" dataKey="ch4" stroke="#f59e0b" dot={false} strokeWidth={1.5} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Center: Probability Bars */}
        <div className="col-span-1 bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold mb-6 text-gray-800 text-center">Motor Function Probability</h2>
          <div className="space-y-6">
            {MOTOR_CLASSES.map((cls, idx) => (
              <div key={idx}>
                <div className="flex justify-between mb-2">
                  <span className="font-semibold text-gray-700">{cls.name}</span>
                  <span className="font-bold" style={{ color: cls.color }}>
                    {(probabilities[idx] * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-8 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-300 flex items-center justify-end pr-2"
                    style={{
                      width: `${probabilities[idx] * 100}%`,
                      backgroundColor: cls.color
                    }}
                  >
                    {probabilities[idx] > 0.15 && (
                      <span className="text-white text-xs font-bold">
                        {(probabilities[idx] * 100).toFixed(0)}%
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

        {/* Right: Radar Chart */}
        <div className="col-span-1 bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold mb-4 text-gray-800 text-center">Confidence Radar</h2>
          <ResponsiveContainer width="100%" height="90%">
            <RadarChart data={radarData}>
              <PolarGrid stroke="#e5e7eb" />
              <PolarAngleAxis dataKey="class" tick={{ fontSize: 12, fill: '#374151' }} />
              <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 10 }} />
              <Radar
                name="Confidence %"
                dataKey="confidence"
                stroke="#8b5cf6"
                fill="#8b5cf6"
                fillOpacity={0.6}
                strokeWidth={2}
              />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default EEGSimulator;

