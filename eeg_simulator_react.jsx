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
  const [useValidationData, setUseValidationData] = useState(false);
  const [validationDataLoaded, setValidationDataLoaded] = useState(false);
  const [sampleInfo, setSampleInfo] = useState(null);
  const timeRef = useRef(0);
  const maxDataPoints = 100;
  const API_URL = 'http://localhost:8000';
  // Update this path to match your data location
  const DATA_PATH = '/Users/anushkaarjun/Desktop/Outside of School/Prosethic Research Data/files 2';

  // Generate realistic EEG signal matching Python preprocessing
  const generateEEGSignal = (t, freq, amp, noise) => {
    // Multi-frequency signal similar to real EEG (8-30 Hz band)
    const base = amp * Math.sin(2 * Math.PI * freq * t);
    const alpha = 0.3 * amp * Math.sin(2 * Math.PI * (freq * 0.8) * t);
    const beta = 0.2 * amp * Math.sin(2 * Math.PI * (freq * 1.2) * t);
    const noise_component = (Math.random() - 0.5) * noise;
    return base + alpha + beta + noise_component;
  };

  // Check API health and load validation data
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
        
        // Try to load validation data
        try {
          const valResponse = await fetch(`${API_URL}/load_validation_data`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              base_path: DATA_PATH,
              max_subjects: 5
            })
          });
          const valData = await valResponse.json();
          if (valData.status === 'success') {
            setValidationDataLoaded(true);
            setUseValidationData(true);
            console.log(`Loaded ${valData.samples_loaded} validation samples`);
          }
        } catch (valError) {
          console.log('Validation data not available, using simulation');
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

      let channels = [];
      let displayedChannels = [];
      let newProbs = MOTOR_CLASSES.map(() => 0.2);
      let actualLabel = '';
      let actualIdx = 0;

      // Use validation data if available
      if (useValidationData && validationDataLoaded) {
        try {
          const response = await fetch(`${API_URL}/get_validation_sample`);
          const data = await response.json();
          
          // Extract channels (64 channels, 125 samples each)
          channels = data.channels;
          
          // Get first 4 channels for display - each channel is an array of 125 samples
          // We'll display the time series by creating data points for each sample
          if (channels.length >= 4 && channels[0].length > 0) {
            // Create time series data points from the 125 samples
            const sampleRate = 250; // Hz
            const timeStep = 1 / sampleRate; // 0.004 seconds per sample
            
            // Update EEG display with all 125 samples from first 4 channels
            const timeSeriesData = [];
            for (let i = 0; i < Math.min(125, channels[0].length); i++) {
              const sampleTime = i * timeStep;
              timeSeriesData.push({
                time: sampleTime.toFixed(3),
                ch1: channels[0][i] || 0,
                ch2: channels[1][i] || 0,
                ch3: channels[2][i] || 0,
                ch4: channels[3][i] || 0
              });
            }
            
            // Update the full time series (replace old data)
            setEegData(timeSeriesData.slice(-maxDataPoints));
            
            // For display channels, use the last sample
            displayedChannels = [
              channels[0][channels[0].length - 1] || 0,
              channels[1][channels[1].length - 1] || 0,
              channels[2][channels[2].length - 1] || 0,
              channels[3][channels[3].length - 1] || 0
            ];
          }
          
          // Get probabilities from prediction
          if (data.probabilities) {
            // Map API classes to our MOTOR_CLASSES order
            const apiClasses = data.classes || [];
            const classMap = {};
            apiClasses.forEach((cls, idx) => {
              const ourIdx = MOTOR_CLASSES.findIndex(mc => mc.name === cls);
              if (ourIdx >= 0) {
                classMap[idx] = ourIdx;
              }
            });
            
            // Reorder probabilities to match MOTOR_CLASSES
            newProbs = MOTOR_CLASSES.map((_, idx) => {
              const apiIdx = apiClasses.findIndex(ac => ac === MOTOR_CLASSES[idx].name);
              return apiIdx >= 0 ? data.probabilities[apiIdx] : 0.2;
            });
            
            // Normalize
            const sum = newProbs.reduce((a, b) => a + b, 0);
            if (sum > 0) {
              newProbs = newProbs.map(p => p / sum);
            }
          }
          
          // Set actual class from validation data
          actualLabel = data.actual_label || '';
          actualIdx = MOTOR_CLASSES.findIndex(mc => mc.name === actualLabel);
          if (actualIdx < 0) actualIdx = 0;
          setActualClass(actualIdx);
          
          // Store sample info
          setSampleInfo({
            index: data.sample_index,
            total: data.total_samples
          });
          
        } catch (error) {
          console.error('Error fetching validation sample:', error);
          // Fall back to simulation
        }
      }
      
      // Fallback to simulation if validation data not available
      if (!useValidationData || !validationDataLoaded || channels.length === 0) {
        // Generate multi-channel EEG data
        channels = [];
        displayedChannels = [];
        
        for (let ch = 0; ch < 64; ch++) {
          const freq = 8 + (ch % 10) * 2.2;
          const amp = 40 + Math.random() * 20;
          const signal = generateEEGSignal(t, freq, amp, 15);
          // Create array of 125 samples (simulating 0.5s at 250Hz)
          const channelSamples = Array(125).fill(0).map((_, i) => 
            generateEEGSignal(t + i * 0.004, freq, amp, 15)
          );
          channels.push(channelSamples);
          
          if (ch < 4) {
            displayedChannels.push(signal);
          }
        }
        
        // Simulated probabilities
        newProbs = MOTOR_CLASSES.map((_, idx) => {
          const base = Math.random() * 0.3;
          const boost = idx === actualClass ? 0.5 + Math.random() * 0.3 : 0;
          return Math.min(base + boost, 1);
        });
        
        const sum = newProbs.reduce((a, b) => a + b, 0);
        newProbs = newProbs.map(p => p / sum);
        
        // Randomly change actual class
        if (Math.random() < 0.01) {
          actualIdx = Math.floor(Math.random() * MOTOR_CLASSES.length);
          setActualClass(actualIdx);
        } else {
          actualIdx = actualClass;
        }
      }

      // Update EEG display data
      // For validation data, we need to extract time series from the channel samples
      // For now, use the current sample value or generate if not available
      const newPoint = {
        time: t.toFixed(2),
        ch1: displayedChannels[0] !== undefined ? displayedChannels[0] : generateEEGSignal(t, 10, 50, 20),
        ch2: displayedChannels[1] !== undefined ? displayedChannels[1] : generateEEGSignal(t, 12, 45, 18),
        ch3: displayedChannels[2] !== undefined ? displayedChannels[2] : generateEEGSignal(t, 8, 55, 22),
        ch4: displayedChannels[3] !== undefined ? displayedChannels[3] : generateEEGSignal(t, 15, 40, 15)
      };

      // Only update if not using validation data (validation data updates separately)
      if (!useValidationData || !validationDataLoaded) {
        setEegData(prev => {
          const updated = [...prev, newPoint];
          return updated.slice(-maxDataPoints);
        });
      }

      setProbabilities(newProbs);

      // Determine predicted class
      const maxProb = Math.max(...newProbs);
      const predicted = newProbs.indexOf(maxProb);
      setPredictedClass(predicted);
      setIsCorrect(predicted === actualIdx);
      
    }, 200); // Slower update for validation data (200ms = 5 samples/second)

    return () => clearInterval(interval);
  }, [actualClass, useAPI, apiStatus, useValidationData, validationDataLoaded]);

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
      <div className="absolute top-4 right-4 z-10 flex flex-col gap-2">
        <div className={`${getApiStatusColor()} text-white rounded-lg px-4 py-2 shadow-lg text-sm font-semibold`}>
          {getApiStatusText()}
        </div>
        {validationDataLoaded && (
          <div className="bg-blue-500 text-white rounded-lg px-4 py-2 shadow-lg text-sm font-semibold">
            Using Real Validation Data
            {sampleInfo && (
              <div className="text-xs mt-1">
                Sample {sampleInfo.index + 1} / {sampleInfo.total}
              </div>
            )}
          </div>
        )}
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

