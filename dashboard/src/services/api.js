import { useState, useEffect } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

// Initial base state for mock data
let mockDriverState = {
  risk_score: 2,
  risk_level: "SAFE",
  speed: 62,
  ear: 0.28,
  emotion: "neutral",
  phone_detected: false,
  passenger_detected: false,
  talking_detected: false,
  driver_distracted: false,
  alerts_today: 4
};

// Generate realistic simulated fluctuations
const updateMockState = () => {
  const time = new Date().toLocaleTimeString();
  
  // High-probability states remain normal, but occasionally simulate risks
  const rand = Math.random();
  
  if (rand < 0.05) {
    // Simulate distraction
    mockDriverState.driver_distracted = true;
    mockDriverState.risk_score = Math.min(20, mockDriverState.risk_score + 4);
  } else if (rand < 0.10) {
    // Simulate phone use
    mockDriverState.phone_detected = true;
    mockDriverState.risk_score = Math.min(20, mockDriverState.risk_score + 7);
  } else if (rand < 0.15) {
    // Simulate drowsiness (Low EAR)
    mockDriverState.ear = parseFloat((0.14 + Math.random() * 0.05).toFixed(2));
    mockDriverState.risk_score = Math.min(20, mockDriverState.risk_score + 5);
  } else if (rand < 0.20) {
    // Simulate excessive talking
    mockDriverState.talking_detected = true;
    mockDriverState.risk_score = Math.min(20, mockDriverState.risk_score + 3);
  } else if (rand > 0.85) {
    // Recovery to safe state
    mockDriverState.driver_distracted = false;
    mockDriverState.phone_detected = false;
    mockDriverState.talking_detected = false;
    mockDriverState.ear = parseFloat((0.25 + Math.random() * 0.08).toFixed(2));
    mockDriverState.risk_score = Math.max(1, mockDriverState.risk_score - 3);
  } else {
    // Minor normal fluctuations
    mockDriverState.ear = parseFloat((0.24 + Math.random() * 0.06).toFixed(2));
    mockDriverState.speed = Math.max(0, Math.min(120, mockDriverState.speed + Math.floor(Math.random() * 7) - 3));
  }

  // Handle speed-fatigue multiplier
  if (mockDriverState.speed > 80 && (mockDriverState.ear < 0.20 || mockDriverState.driver_distracted)) {
    mockDriverState.risk_score = Math.min(20, mockDriverState.risk_score + 6);
  }

  // Recalculate level
  if (mockDriverState.risk_score >= 11) {
    mockDriverState.risk_level = "HIGH RISK";
    mockDriverState.emotion = Math.random() > 0.5 ? "fear" : "sad";
  } else if (mockDriverState.risk_score >= 6) {
    mockDriverState.risk_level = "WARNING";
    mockDriverState.emotion = Math.random() > 0.5 ? "surprise" : "anger";
  } else {
    mockDriverState.risk_level = "SAFE";
    mockDriverState.emotion = Math.random() > 0.8 ? "happy" : "neutral";
  }

  // Generated active alert
  let activeAlerts = [];
  if (mockDriverState.ear < 0.20) {
    activeAlerts.push({
      time,
      type: "Drowsiness Alert",
      severity: "High"
    });
  }
  if (mockDriverState.phone_detected) {
    activeAlerts.push({
      time,
      type: "Phone Usage",
      severity: "High"
    });
  }
  if (mockDriverState.driver_distracted) {
    activeAlerts.push({
      time,
      type: "Gaze Distraction",
      severity: "Medium"
    });
  }
  if (mockDriverState.talking_detected) {
    activeAlerts.push({
      time,
      type: "Excessive Talking",
      severity: "Low"
    });
  }

  return {
    ...mockDriverState,
    alerts: activeAlerts
  };
};

export const fetchDriverStatus = async () => {
  try {
    // Attempt FastAPI Call
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1000); // 1s timeout
    
    const response = await fetch(`${API_BASE_URL}/driver-status`, { signal: controller.signal });
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return { data, isMock: false };
  } catch (error) {
    // Failover silently to simulated mock state
    const data = updateMockState();
    return { data, isMock: true };
  }
};
