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
  const [feedMode, setFeedMode] = useState('browser'); // Default to browser mode for cloud deployments
  const [demoControls, setDemoControls] = useState({
    phone_detected: false,
    passenger_detected: false,
    speed: 61,
  });

  const [driverData, setDriverData] = useState({
    risk_score: 0,
    risk_level: "SAFE",
    speed: 61,
    ear: 0.28,
    emotion: "neutral",
    phone_detected: false,
    passenger_detected: false,
    talking_detected: false,
    driver_distracted: false,
    alerts: []
  });

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

  const lastAlertTimes = useRef({});

  const tripStartTime = useRef(Date.now());

  const handleBrowserTelemetryUpdate = (telemetry) => {
    if (feedMode !== 'browser') return;
    setDriverData(prev => {
      const updated = {
        ...prev,
        ...telemetry,
        phone_detected: demoControls.phone_detected,
        passenger_detected: demoControls.passenger_detected,
        speed: demoControls.speed,
      };

      // Calculate risk score based on active metrics
      let riskScore = 0;
      if (updated.ear < 0.20) riskScore += 5; 
      if (updated.driver_distracted) riskScore += 4; 
      if (updated.phone_detected) riskScore += 7; 
      if (updated.talking_detected) riskScore += 3; 
      if (updated.passenger_detected) riskScore += 3; 

      if (updated.speed > 80 && (updated.ear < 0.20 || updated.driver_distracted)) {
        riskScore += 6;
      }

      updated.risk_score = Math.min(20, riskScore);

      if (updated.risk_score >= 11) {
        updated.risk_level = "HIGH RISK";
      } else if (updated.risk_score >= 6) {
        updated.risk_level = "WARNING";
      } else {
        updated.risk_level = "SAFE";
      }

      return updated;
    });
  };

  // Dynamic Polling Loop
  useEffect(() => {
    let active = true;

    const poll = async () => {
      try {
        let isBackendOnline = false;
        let backendData = null;

        try {
          const { data, isMock } = await fetchDriverStatus();
          isBackendOnline = !isMock;
          if (feedMode === 'backend') {
            backendData = data;
          }
        } catch (e) {
          isBackendOnline = false;
        }

        if (!active) return;
        setIsOnline(isBackendOnline);

        if (feedMode === 'backend' && backendData) {
          setDriverData(backendData);
          setError(null);
        }

        setLoading(false);

        // Update charts and histories based on current driverData state
        setDriverData(prev => {
          if (!prev) return prev;

          const timeString = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
          
          // Update Risk history
          setRiskHistory(prevRisk => {
            const updated = [...prevRisk, { time: timeString, risk_score: prev.risk_score }];
            return updated.slice(-12);
          });

          // Update talking volume levels
          setTalkingHistory(prevTalking => {
            const decibel = prev.talking_detected ? (50 + Math.floor(Math.random() * 30)) : (5 + Math.floor(Math.random() * 10));
            const updated = [...prevTalking, { time: timeString, level: decibel }];
            return updated.slice(-12);
          });

          // Update Cumulative Emotion counters
          setEmotionStats(prevEmotion => ({
            ...prevEmotion,
            [prev.emotion]: (prevEmotion[prev.emotion] || 0) + 1
          }));

          // Incremental incident counters logic
          let phoneInc = 0;
          let drowsyInc = 0;
          let distractInc = 0;

          const isDrowsyEAR = prev.ear < 0.20;

          if (prev.phone_detected && !prevStates.current.phone) {
            phoneInc = 1;
          }
          if (isDrowsyEAR && !prevStates.current.drowsy) {
            drowsyInc = 1;
          }
          if (prev.driver_distracted && !prevStates.current.distracted) {
            distractInc = 1;
          }

          prevStates.current = {
            phone: prev.phone_detected,
            drowsy: isDrowsyEAR,
            distracted: prev.driver_distracted
          };

          const deductions = (phoneInc * 8) + (drowsyInc * 6) + (distractInc * 4);

          setTripSummary(prevTrip => {
            const elapsedMs = Date.now() - tripStartTime.current;
            const mins = Math.floor(elapsedMs / 60000);
            const hrs = Math.floor(mins / 60);
            const driveTimeString = `${hrs}h ${String(mins % 60).padStart(2, '0')}m`;

            return {
              safety_score: Math.max(40, prevTrip.safety_score - deductions),
              drive_time: driveTimeString,
              drowsy_count: prevTrip.drowsy_count + drowsyInc,
              phone_count: prevTrip.phone_count + phoneInc,
              distraction_count: prevTrip.distraction_count + distractInc
            };
          });

          if (drowsyInc > 0 || distractInc > 0) {
            setEventHistory(prevEvent => {
              const currentHour = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false }).split(':')[0] + ':00';
              const index = prevEvent.findIndex(item => item.hour === currentHour);
              if (index !== -1) {
                const updated = [...prevEvent];
                updated[index] = {
                  ...updated[index],
                  Drowsiness: updated[index].Drowsiness + drowsyInc,
                  Distraction: updated[index].Distraction + distractInc
                };
                return updated;
              } else {
                const updated = [...prevEvent, { hour: currentHour, Drowsiness: drowsyInc, Distraction: distractInc }];
                return updated.slice(-6);
              }
            });
          }

          // Browser Mode Alert Logs & TTS Speech Synthesis
          if (feedMode === 'browser') {
            const currentAlerts = [];
            const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

            const speedDrowsyCond = (isDrowsyEAR && prev.speed > 80);
            const phoneFatigueCond = (prev.phone_detected && isDrowsyEAR);

            if (speedDrowsyCond) {
              currentAlerts.push({
                type: "SPEED_DROWSY",
                title: "Speed Drowsiness",
                severity: "High",
                message: "Danger! High speed and fatigue detected. SLOW DOWN IMMEDIATELY!"
              });
            } else if (isDrowsyEAR) {
              currentAlerts.push({
                type: "DROWSY",
                title: "Drowsiness Alert",
                severity: "High",
                message: "Driver is drowsy. Please stay alert."
              });
            }

            if (phoneFatigueCond) {
              currentAlerts.push({
                type: "PHONE_AND_FATIGUE",
                title: "Critical Drowsy Phone",
                severity: "High",
                message: "Critical warning! You are tired and using a phone. Please stop immediately."
              });
            } else if (prev.phone_detected) {
              currentAlerts.push({
                type: "PHONE",
                title: "Phone Usage",
                severity: "High",
                message: "Phone usage detected. Please focus on driving."
              });
            }

            if (prev.driver_distracted && !isDrowsyEAR) {
              currentAlerts.push({
                type: "LOOKING_AWAY",
                title: "Gaze Distraction",
                severity: "Medium",
                message: "Please focus on driving."
              });
            }

            if (prev.talking_detected) {
              currentAlerts.push({
                type: "TALKING",
                title: "Excessive Talking",
                severity: "Low",
                message: "Excessive talking detected. Focus on driving."
              });
            }

            if (currentAlerts.length > 0) {
              setAlertsLog(prevAlerts => {
                const now = Date.now();
                const newAlerts = [];

                currentAlerts.forEach(alertItem => {
                  const lastTriggered = lastAlertTimes.current[alertItem.type] || 0;
                  const cooldown = alertItem.type === 'TALKING' ? 10000 : 5000; // Cooldown matching python backend

                  if (now - lastTriggered >= cooldown) {
                    lastAlertTimes.current[alertItem.type] = now;
                    newAlerts.push({
                      time: timestamp,
                      type: alertItem.title,
                      severity: alertItem.severity,
                      message: alertItem.message
                    });

                    // TTS Speech Synthesis Voice Warn
                    if (typeof window !== 'undefined' && window.speechSynthesis) {
                      const utterance = new SpeechSynthesisUtterance(alertItem.message);
                      utterance.rate = 0.95;
                      window.speechSynthesis.speak(utterance);
                    }
                  }
                });

                if (newAlerts.length > 0) {
                  return [...newAlerts, ...prevAlerts].slice(0, 15);
                }
                return prevAlerts;
              });
            }
          } else {
            // Backend Mode Alert Logs
            if (prev.alerts && prev.alerts.length > 0) {
              setAlertsLog(prevAlerts => {
                const newAlerts = prev.alerts.filter(
                  item => !prevAlerts.some(logged => logged.time === item.time && logged.type === item.type)
                );
                return [...newAlerts, ...prevAlerts].slice(0, 15);
              });
            }
          }

          return prev;
        });

      } catch (err) {
        if (active) setError(err.message);
      }
    };

    poll();
    const interval = setInterval(poll, 2000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [feedMode]);

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
      <Header isOnline={isOnline} feedMode={feedMode} setFeedMode={setFeedMode} />

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
      {error && feedMode === 'backend' && (
        <div className="bg-red-500/15 border border-red-500/30 text-red-400 p-3 rounded-2xl mb-6 flex items-center justify-between text-xs font-semibold">
          <span>Failed to synchronize live sensor feeds: {error}</span>
          <span className="bg-red-500/20 px-2 py-0.5 rounded border border-red-500/30">Local Offline State</span>
        </div>
      )}

      {/* 2. KPI cards block */}
      <KPICards data={{ ...driverData, alerts_today: tripSummary.drowsy_count + tripSummary.phone_count + tripSummary.distraction_count + 1 }} />

      {/* 3. Live Monitoring and Feed */}
      <LiveMonitoring 
        data={driverData} 
        feedMode={feedMode}
        demoControls={demoControls}
        setDemoControls={setDemoControls}
        onBrowserTelemetryUpdate={handleBrowserTelemetryUpdate}
      />

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
