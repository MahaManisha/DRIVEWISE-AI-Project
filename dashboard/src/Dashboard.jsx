import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import KPICards from './components/KPICards';
import LiveMonitoring from './components/LiveMonitoring';
import RiskAnalysis from './components/RiskAnalysis';
import AttentionAnalytics from './components/AttentionAnalytics';
import AlertCenter from './components/AlertCenter';
import SafetySummary from './components/SafetySummary';
import Footer from './components/Footer';
import { fetchDriverStatus } from './services/api';
import { ShieldAlert, RefreshCw, RefreshCwOff } from 'lucide-react';

export default function Dashboard() {
  const [driverData, setDriverData] = useState(null);
  const [isOnline, setIsOnline] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Historical chart states (keep last 12 readings)
  const [riskHistory, setRiskHistory] = useState([]);
  const [talkingHistory, setTalkingHistory] = useState([]);
  const [eventHistory, setEventHistory] = useState([
    { hour: '10:00', Drowsiness: 1, Distraction: 2 },
    { hour: '11:00', Drowsiness: 0, Distraction: 4 },
    { hour: '12:00', Drowsiness: 2, Distraction: 1 },
    { hour: '13:00', Drowsiness: 3, Distraction: 5 },
    { hour: '14:00', Drowsiness: 1, Distraction: 2 },
    { hour: '15:00', Drowsiness: 2, Distraction: 3 }
  ]);
  
  // Real-time alerts log table
  const [alertsLog, setAlertsLog] = useState([]);
  
  // Emotion tracker counters
  const [emotionStats, setEmotionStats] = useState({
    neutral: 15,
    happy: 25,
    sad: 4,
    fear: 2,
    anger: 1,
    surprise: 3
  });

  // Trip safety parameters state
  const [tripSummary, setTripSummary] = useState({
    safety_score: 96,
    drive_time: "0h 02m",
    drowsy_count: 1,
    phone_count: 1,
    distraction_count: 2
  });

  // Track previous states to trigger incident counter increases only once per event cycle
  const prevStates = useRef({
    phone: false,
    drowsy: false,
    distracted: false
  });

  const tripStartTime = useRef(Date.now());

  // Dynamic Polling Loop
  useEffect(() => {
    let active = true;

    const poll = async () => {
      try {
        const { data, isMock } = await fetchDriverStatus();
        if (!active) return;

        setDriverData(data);
        setIsOnline(!isMock);
        setError(null);
        setLoading(false);

        const timeString = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const timeMinute = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        // Update Risk history
        setRiskHistory(prev => {
          const updated = [...prev, { time: timeString, risk_score: data.risk_score }];
          return updated.slice(-12); // keep last 12 points
        });

        // Update talking volume levels
        setTalkingHistory(prev => {
          const decibel = data.talking_detected ? (50 + Math.floor(Math.random() * 30)) : (5 + Math.floor(Math.random() * 10));
          const updated = [...prev, { time: timeString, level: decibel }];
          return updated.slice(-12);
        });

        // Update Cumulative Emotion counters
        setEmotionStats(prev => ({
          ...prev,
          [data.emotion]: (prev[data.emotion] || 0) + 1
        }));

        // Incremental incident counters logic
        let phoneInc = 0;
        let drowsyInc = 0;
        let distractInc = 0;

        const isDrowsyEAR = data.ear < 0.20;

        // Detect transitions (false -> true)
        if (data.phone_detected && !prevStates.current.phone) {
          phoneInc = 1;
        }
        if (isDrowsyEAR && !prevStates.current.drowsy) {
          drowsyInc = 1;
        }
        if (data.driver_distracted && !prevStates.current.distracted) {
          distractInc = 1;
        }

        // Save current state as previous
        prevStates.current = {
          phone: data.phone_detected,
          drowsy: isDrowsyEAR,
          distracted: data.driver_distracted
        };

        // Calculate Safety Score deduction
        const deductions = (phoneInc * 8) + (drowsyInc * 6) + (distractInc * 4);

        // Update Safety Summary Panel
        setTripSummary(prev => {
          // Format elapsed duration
          const elapsedMs = Date.now() - tripStartTime.current;
          const mins = Math.floor(elapsedMs / 60000);
          const hrs = Math.floor(mins / 60);
          const driveTimeString = `${hrs}h ${String(mins % 60).padStart(2, '0')}m`;

          return {
            safety_score: Math.max(40, prev.safety_score - deductions),
            drive_time: driveTimeString,
            drowsy_count: prev.drowsy_count + drowsyInc,
            phone_count: prev.phone_count + phoneInc,
            distraction_count: prev.distraction_count + distractInc
          };
        });

        // Map hourly counts to Cognitive Anomalies chart
        if (drowsyInc > 0 || distractInc > 0) {
          setEventHistory(prev => {
            const currentHour = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false }).split(':')[0] + ':00';
            const index = prev.findIndex(item => item.hour === currentHour);
            if (index !== -1) {
              const updated = [...prev];
              updated[index] = {
                ...updated[index],
                Drowsiness: updated[index].Drowsiness + drowsyInc,
                Distraction: updated[index].Distraction + distractInc
              };
              return updated;
            } else {
              const updated = [...prev, { hour: currentHour, Drowsiness: drowsyInc, Distraction: distractInc }];
              return updated.slice(-6);
            }
          });
        }

        // Push new alerts to Alert Center table
        if (data.alerts && data.alerts.length > 0) {
          setAlertsLog(prev => {
            // Filter duplicates by checking time and alert type
            const newAlerts = data.alerts.filter(
              item => !prev.some(logged => logged.time === item.time && logged.type === item.type)
            );
            return [...newAlerts, ...prev].slice(0, 15); // keep max 15 alerts in table
          });
        }

      } catch (err) {
        if (active) setError(err.message);
      }
    };

    poll(); // first immediately
    const interval = setInterval(poll, 2000); // Poll every 2 seconds

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  // Alert resolve action handler
  const handleAcknowledge = (index) => {
    setAlertsLog(prev => prev.filter((_, idx) => idx !== index));
  };

  const handleMute = () => {
    alert("Audio TTS warnings temporarily muted.");
  };

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-[#0b0f19] text-white">
        <div className="flex items-center gap-3">
          <RefreshCw className="w-8 h-8 text-blue-400 animate-spin" />
          <h2 className="text-lg font-bold tracking-wider font-mono">LOADING SYSTEM telemetry...</h2>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-[1440px] mx-auto p-4 md:p-6 select-none">
      
      {/* 1. Header component */}
      <Header isOnline={isOnline} />

      {/* Warning Toast Banner if high risk */}
      {driverData?.risk_level === 'HIGH RISK' && (
        <div className="bg-red-500/10 border-2 border-red-500/30 text-red-400 p-4 rounded-2xl mb-6 flex items-center gap-3 shadow-[0_0_20px_rgba(239,68,68,0.2)] animate-pulse-slow">
          <ShieldAlert className="w-6 h-6 flex-shrink-0 animate-bounce" />
          <div>
            <h4 className="font-bold text-sm">CRITICAL COGNITIVE RISK DETECTED</h4>
            <p className="text-xs opacity-90 mt-0.5">Please ensure driver focus. Immediate rest stop or passenger relief required.</p>
          </div>
        </div>
      )}

      {/* Error alert banner */}
      {error && (
        <div className="bg-red-500/15 border border-red-500/30 text-red-400 p-3 rounded-2xl mb-6 flex items-center justify-between text-xs font-semibold">
          <span>Failed to synchronize live sensor feeds: {error}</span>
          <span className="bg-red-500/20 px-2 py-0.5 rounded border border-red-500/30">Local Offline State</span>
        </div>
      )}

      {/* 2. KPI cards block */}
      <KPICards data={{ ...driverData, alerts_today: tripSummary.drowsy_count + tripSummary.phone_count + tripSummary.distraction_count + 1 }} />

      {/* 3. Live Monitoring and Feed */}
      <LiveMonitoring data={driverData} />

      {/* 4. Risk Analysis (Gauge & Trend) */}
      <RiskAnalysis 
        currentScore={driverData.risk_score} 
        level={driverData.risk_level} 
        history={riskHistory} 
      />

      {/* 5. Driver Health & Attention Analytics */}
      <AttentionAnalytics 
        emotionStats={emotionStats} 
        eventHistory={eventHistory} 
        talkingHistory={talkingHistory} 
      />

      {/* 6. Alert Center table */}
      <AlertCenter 
        alerts={alertsLog} 
        onAcknowledge={handleAcknowledge}
        onMute={handleMute}
      />

      {/* 7. Safety Summary Panel */}
      <SafetySummary summary={tripSummary} />

      {/* 8. Footer component */}
      <Footer />

    </div>
  );
}
