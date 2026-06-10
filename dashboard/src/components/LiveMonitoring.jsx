import React, { useState } from 'react';
import { 
  Camera, PhoneOff, PhoneCall, Users, Users2, 
  Compass, MessageSquareCode, MessageSquareOff 
} from 'lucide-react';

export default function LiveMonitoring({ data }) {
  const {
    phone_detected = false,
    passenger_detected = false,
    talking_detected = false,
    driver_distracted = false,
    emotion = 'neutral',
    ear = 0.28
  } = data;

  const [streamError, setStreamError] = useState(false);

  const isDrowsy = ear < 0.20;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
      
      {/* Live Video Feed - Left Column (takes 2 spaces on desktop) */}
      <div className="lg:col-span-2 glass-card rounded-2xl p-4 flex flex-col justify-between overflow-hidden relative min-h-[350px]">
        {/* Top bar indicators */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3 z-10">
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" />
            <span className="text-xs font-mono font-bold tracking-wider text-slate-300">
              {streamError ? 'OFFLINE SIMULATION // COGNITIVE SCAN' : 'LIVE VEHICLE STREAM // COGNITIVE SCAN'}
            </span>
          </div>
          <div className="flex items-center gap-3 text-[10px] font-mono text-slate-400">
            <span>FPS: <strong className="text-emerald-400">29.8</strong></span>
            <span>RES: <strong>1280x720</strong></span>
            <span>SYS_TEMP: <strong>44°C</strong></span>
          </div>
        </div>

        {/* Camera Visualizer Screen */}
        <div className="relative flex-1 rounded-xl bg-slate-950 overflow-hidden flex items-center justify-center border border-slate-900 min-h-[260px]">
          
          {/* Real MJPEG Video Stream from Python server */}
          {!streamError ? (
            <img 
              src="http://localhost:8000/api/video-feed" 
              alt="Live Driver Stream"
              className="absolute inset-0 w-full h-full object-cover"
              onError={() => setStreamError(true)}
            />
          ) : (
            /* Fallback Mock Camera Icon if offline */
            <Camera className="w-12 h-12 text-slate-800 opacity-20 pointer-events-none" />
          )}

          {/* Static Scanline Overlay - only visible if mockup is showing */}
          {streamError && (
            <>
              <div className="absolute inset-0 bg-linear-to-b from-transparent via-blue-500/2 to-transparent pointer-events-none" />
              <div className="absolute left-0 right-0 h-[2px] bg-blue-500/30 shadow-lg animate-scan pointer-events-none" />
              <div className="absolute inset-0 bg-[radial-gradient(#1e293b_1px,transparent_1px)] [background-size:16px_16px] opacity-40" />
            </>
          )}

          {/* AI Face Landmarks Tracker Simulation - only visible when offline / streamError is true */}
          {streamError && (
            <div className="absolute inset-0 flex items-center justify-center">
              {/* Outer Face Bounding Box */}
              <div className={`w-48 h-56 border-2 rounded-3xl flex flex-col justify-between p-3 transition-all duration-300 ${isDrowsy ? 'border-red-500 shadow-[0_0_30px_rgba(239,68,68,0.2)]' : driver_distracted ? 'border-amber-400 shadow-[0_0_30px_rgba(234,179,8,0.2)]' : 'border-blue-500 shadow-[0_0_30px_rgba(59,130,246,0.15)]'}`}>
                <div className="flex justify-between items-start">
                  <span className="text-[9px] font-mono bg-blue-500/20 text-blue-300 px-1.5 py-0.5 rounded border border-blue-500/30 uppercase leading-none">
                    Face_01
                  </span>
                  <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded border leading-none uppercase ${isDrowsy ? 'bg-red-500/20 text-red-300 border-red-500/30' : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'}`}>
                    {isDrowsy ? 'Fatigue' : 'Active'}
                  </span>
                </div>

                {/* Eye Landmark Targets */}
                <div className="flex justify-around items-center my-auto w-full px-4">
                  <div className={`w-6 h-6 border-2 border-dashed rounded-full flex items-center justify-center ${isDrowsy ? 'border-red-400 animate-pulse' : 'border-blue-400'}`}>
                    <div className={`w-1.5 h-1.5 rounded-full ${isDrowsy ? 'bg-red-500' : 'bg-blue-400'}`} />
                  </div>
                  <div className={`w-6 h-6 border-2 border-dashed rounded-full flex items-center justify-center ${isDrowsy ? 'border-red-400 animate-pulse' : 'border-blue-400'}`}>
                    <div className={`w-1.5 h-1.5 rounded-full ${isDrowsy ? 'bg-red-500' : 'bg-blue-400'}`} />
                  </div>
                </div>

                {/* Bottom readouts */}
                <div className="flex justify-between items-end text-[9px] font-mono text-slate-400 leading-none">
                  <span>EAR: {(ear).toFixed(2)}</span>
                  <span>MOOD: {emotion.toUpperCase()}</span>
                </div>
              </div>

              {/* Simulated Phone Bounding Box if detected */}
              {phone_detected && (
                <div className="absolute right-12 bottom-12 w-28 h-36 border-2 border-red-500 bg-red-950/20 rounded-lg flex flex-col justify-between p-2 animate-pulse shadow-[0_0_20px_rgba(239,68,68,0.2)]">
                  <span className="text-[9px] font-mono bg-red-500 text-white px-1 py-0.5 rounded leading-none font-bold uppercase self-start">
                    Mobile Phone
                  </span>
                  <span className="text-[9px] font-mono text-red-400 text-right font-bold leading-none">
                    CONF: 98%
                  </span>
                </div>
              )}

              {/* Simulated Passenger Bounding Box if detected */}
              {passenger_detected && (
                <div className="absolute left-8 bottom-16 w-32 h-44 border-2 border-cyan-500 bg-cyan-950/10 rounded-xl flex flex-col justify-between p-2 shadow-[0_0_15px_rgba(6,182,212,0.15)]">
                  <span className="text-[9px] font-mono bg-cyan-500 text-slate-900 px-1 py-0.5 rounded leading-none font-bold uppercase self-start">
                    Passenger
                  </span>
                  <span className="text-[9px] font-mono text-cyan-400 text-right leading-none">
                    CONF: 94%
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Bottom Status Ribbon */}
        <div className="flex flex-wrap items-center justify-between text-xs text-slate-400 mt-3 border-t border-slate-900 pt-3 gap-2">
          <div className="flex items-center gap-2">
            <span className="inline-block w-2 h-2 rounded-full bg-blue-500" />
            <span>AI Models: MediaPipe FaceMesh V2, YOLOv8n</span>
          </div>
          <span>Camera Index: 0 (USB Webcam)</span>
        </div>
      </div>

      {/* Sensor Feeds Details - Right Column */}
      <div className="flex flex-col gap-4">
        
        {/* Phone Detection Status Card */}
        <div className={`glass-card p-4 rounded-2xl flex items-center justify-between border ${phone_detected ? 'border-red-500/30 bg-red-500/5' : 'border-transparent'}`}>
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-xl border ${phone_detected ? 'bg-red-500/20 border-red-500/30 text-red-400 animate-bounce' : 'bg-slate-800/50 border-slate-700/50 text-slate-400'}`}>
              {phone_detected ? <PhoneCall className="w-5 h-5" /> : <PhoneOff className="w-5 h-5" />}
            </div>
            <div>
              <h4 className="text-sm font-semibold text-white">Phone Detection</h4>
              <p className="text-xs text-slate-400 mt-0.5">YOLOv8 Class 67 Monitor</p>
            </div>
          </div>
          <span className={`text-xs font-mono font-bold px-2 py-1 rounded-lg uppercase ${phone_detected ? 'bg-red-500/20 text-red-400 text-neon-red' : 'bg-slate-800 text-slate-400'}`}>
            {phone_detected ? 'Detected' : 'Clear'}
          </span>
        </div>

        {/* Passenger Detection Status Card */}
        <div className={`glass-card p-4 rounded-2xl flex items-center justify-between border ${passenger_detected ? 'border-blue-500/20 bg-blue-500/5' : 'border-transparent'}`}>
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-xl border ${passenger_detected ? 'bg-blue-500/20 border-blue-500/30 text-blue-400' : 'bg-slate-800/50 border-slate-700/50 text-slate-400'}`}>
              {passenger_detected ? <Users className="w-5 h-5" /> : <Users2 className="w-5 h-5" />}
            </div>
            <div>
              <h4 className="text-sm font-semibold text-white">Passenger Presence</h4>
              <p className="text-xs text-slate-400 mt-0.5">Co-pilot/Cabin occupancy</p>
            </div>
          </div>
          <span className={`text-xs font-mono font-bold px-2 py-1 rounded-lg uppercase ${passenger_detected ? 'bg-blue-500/20 text-blue-400' : 'bg-slate-800 text-slate-400'}`}>
            {passenger_detected ? 'Occupied' : 'Vacant'}
          </span>
        </div>

        {/* Head Pose / Looking Away Status Card */}
        <div className={`glass-card p-4 rounded-2xl flex items-center justify-between border ${driver_distracted ? 'border-amber-500/30 bg-amber-500/5' : 'border-transparent'}`}>
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-xl border ${driver_distracted ? 'bg-amber-500/20 border-amber-500/30 text-amber-400 animate-pulse' : 'bg-slate-800/50 border-slate-700/50 text-slate-400'}`}>
              <Compass className={`w-5 h-5 ${driver_distracted ? 'animate-spin' : ''}`} style={{ animationDuration: '3s' }} />
            </div>
            <div>
              <h4 className="text-sm font-semibold text-white">Driver Distraction</h4>
              <p className="text-xs text-slate-400 mt-0.5">Head Pose & Rotation Tracker</p>
            </div>
          </div>
          <span className={`text-xs font-mono font-bold px-2 py-1 rounded-lg uppercase ${driver_distracted ? 'bg-amber-500/20 text-amber-400 text-neon-yellow' : 'bg-slate-800 text-slate-400'}`}>
            {driver_distracted ? 'Looking Away' : 'Focused'}
          </span>
        </div>

        {/* Excessive Talking Card */}
        <div className={`glass-card p-4 rounded-2xl flex items-center justify-between border ${talking_detected ? 'border-amber-500/20 bg-amber-500/5' : 'border-transparent'}`}>
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-xl border ${talking_detected ? 'bg-amber-500/20 border-amber-500/30 text-amber-400' : 'bg-slate-800/50 border-slate-700/50 text-slate-400'}`}>
              {talking_detected ? <MessageSquareCode className="w-5 h-5" /> : <MessageSquareOff className="w-5 h-5" />}
            </div>
            <div>
              <h4 className="text-sm font-semibold text-white">Acoustic Activity</h4>
              <p className="text-xs text-slate-400 mt-0.5">Excessive speaking timeline</p>
            </div>
          </div>
          <span className={`text-xs font-mono font-bold px-2 py-1 rounded-lg uppercase ${talking_detected ? 'bg-amber-500/20 text-amber-400' : 'bg-slate-800 text-slate-400'}`}>
            {talking_detected ? 'Talking' : 'Silent'}
          </span>
        </div>

      </div>

    </div>
  );
}
