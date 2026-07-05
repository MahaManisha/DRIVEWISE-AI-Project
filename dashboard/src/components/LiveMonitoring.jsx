import React, { useState, useEffect, useRef } from 'react';
import { 
  Camera, PhoneOff, PhoneCall, Users, Users2, 
  Compass, MessageSquareCode, MessageSquareOff, Sliders, Mic, MicOff
} from 'lucide-react';

const LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380];

const getDistance = (p1, p2) => {
  return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
};

const calculateEAR = (landmarks, indices) => {
  const p1 = landmarks[indices[0]];
  const p2 = landmarks[indices[1]];
  const p3 = landmarks[indices[2]];
  const p4 = landmarks[indices[3]];
  const p5 = landmarks[indices[4]];
  const p6 = landmarks[indices[5]];

  const v1 = getDistance(p2, p6);
  const v2 = getDistance(p3, p5);
  const h = getDistance(p1, p4);

  if (h === 0) return 0.0;
  return (v1 + v2) / (2.0 * h);
};

// Global cache for MediaPipe instances to avoid reloading when switching modes
let faceLandmarkerPromise = null;
const loadMediaPipe = async () => {
  if (faceLandmarkerPromise) return faceLandmarkerPromise;

  faceLandmarkerPromise = (async () => {
    const vision = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/vision_bundle.mjs");
    const filesetResolver = await vision.FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
    );
    
    return await vision.FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numFaces: 1
    });
  })();

  return faceLandmarkerPromise;
};

export default function LiveMonitoring({ 
  data, 
  feedMode, 
  demoControls, 
  setDemoControls, 
  onBrowserTelemetryUpdate 
}) {
  const {
    phone_detected = false,
    passenger_detected = false,
    talking_detected = false,
    driver_distracted = false,
    emotion = 'neutral',
    ear = 0.28
  } = data;

  const [streamError, setStreamError] = useState(false);
  const [cameraStatus, setCameraStatus] = useState('Loading...');
  const [micStatus, setMicStatus] = useState('Loading...');
  const [talkingFlag, setTalkingFlag] = useState(false);
  const [cameraError, setCameraError] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const faceMissingCount = useRef(0);

  const isDrowsy = ear < 0.20;

  const videoFeedUrl = import.meta.env.VITE_API_BASE_URL
    ? `${import.meta.env.VITE_API_BASE_URL}/video-feed`
    : 'http://localhost:8000/api/video-feed';

  // 1. Microphone Listener Setup (Web Audio API)
  useEffect(() => {
    if (feedMode !== 'browser') {
      setMicStatus('Inactive');
      return;
    }

    let audioContext = null;
    let analyser = null;
    let microphone = null;
    let javascriptNode = null;
    let micStream = null;

    const initMic = async () => {
      try {
        setMicStatus('Requesting access...');
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;

        microphone = audioContext.createMediaStreamSource(micStream);
        javascriptNode = audioContext.createScriptProcessor(2048, 1, 1);

        microphone.connect(analyser);
        analyser.connect(javascriptNode);
        javascriptNode.connect(audioContext.destination);

        let talkingCount = 0;

        javascriptNode.onaudioprocess = () => {
          const array = new Uint8Array(analyser.frequencyBinCount);
          analyser.getByteFrequencyData(array);

          let totalFreqValue = 0;
          for (let i = 0; i < array.length; i++) {
            totalFreqValue += array[i];
          }

          const avgVolume = totalFreqValue / array.length;
          const normalizedVol = avgVolume / 255.0;

          // Stable threshold (0.07) to filter ambient clicks/hums
          if (normalizedVol > 0.07) {
            talkingCount++;
          } else {
            talkingCount = Math.max(0, talkingCount - 2);
          }

          const isCurrentlyTalking = talkingCount > 10;
          setTalkingFlag(isCurrentlyTalking);
        };
        setMicStatus('Active');
      } catch (err) {
        console.warn("Microphone access declined or unavailable:", err);
        setMicStatus('Blocked/None');
        setTalkingFlag(false);
      }
    };

    initMic();

    return () => {
      if (javascriptNode) javascriptNode.disconnect();
      if (microphone) microphone.disconnect();
      if (analyser) analyser.disconnect();
      if (audioContext) audioContext.close();
      if (micStream) {
        micStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [feedMode]);

  // 2. Browser Webcam Feed & MediaPipe Vision Setup
  useEffect(() => {
    if (feedMode !== 'browser') {
      setCameraStatus('Inactive');
      setCameraError(false);
      return;
    }

    let active = true;
    let localStream = null;
    let landmarkerInstance = null;
    let animationFrameId = null;

    const startWebcam = async () => {
      try {
        setCameraStatus('Requesting access...');
        localStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' }
        });

        if (!active) {
          localStream.getTracks().forEach(track => track.stop());
          return;
        }

        if (videoRef.current) {
          videoRef.current.srcObject = localStream;
          videoRef.current.onloadedmetadata = () => {
            if (videoRef.current && active) {
              videoRef.current.play();
            }
          };
        }

        setCameraStatus('Loading FaceMesh AI...');
        landmarkerInstance = await loadMediaPipe();

        if (!active) return;
        setCameraStatus('Active');
        setCameraError(false);

        const processFrame = () => {
          if (!active) return;

          if (videoRef.current && videoRef.current.readyState >= 2 && landmarkerInstance) {
            const video = videoRef.current;
            const canvas = canvasRef.current;

            if (canvas) {
              if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
              }

              const timestamp = performance.now();
              const results = landmarkerInstance.detectForVideo(video, timestamp);

              const ctx = canvas.getContext('2d');
              ctx.clearRect(0, 0, canvas.width, canvas.height);

              let currentEAR = 0.28;
              let distracted = false;

              if (results.faceLandmarks && results.faceLandmarks.length > 0) {
                faceMissingCount.current = 0;
                const landmarks = results.faceLandmarks[0];

                // EAR calculation
                const leftEAR = calculateEAR(landmarks, LEFT_EYE_INDICES);
                const rightEAR = calculateEAR(landmarks, RIGHT_EYE_INDICES);
                currentEAR = (leftEAR + rightEAR) / 2.0;

                // Yaw distraction check (nose to outer corners ratio)
                const dLeft = getDistance(landmarks[1], landmarks[33]);
                const dRight = getDistance(landmarks[1], landmarks[263]);
                const ratio = dLeft / (dRight || 1);
                distracted = (ratio > 1.45 || ratio < 0.69);

                // Draw bounding box
                let minX = 1.0, maxX = 0.0, minY = 1.0, maxY = 0.0;
                landmarks.forEach(lm => {
                  if (lm.x < minX) minX = lm.x;
                  if (lm.x > maxX) maxX = lm.x;
                  if (lm.y < minY) minY = lm.y;
                  if (lm.y > maxY) maxY = lm.y;
                });

                const isCurrentlyDrowsy = currentEAR < 0.20;
                const x = minX * canvas.width;
                const y = minY * canvas.height;
                const w = (maxX - minX) * canvas.width;
                const h = (maxY - minY) * canvas.height;

                ctx.strokeStyle = isCurrentlyDrowsy ? '#ef4444' : (distracted ? '#fbbf24' : '#3b82f6');
                ctx.lineWidth = 3;
                ctx.shadowColor = ctx.strokeStyle;
                ctx.shadowBlur = 12;
                ctx.strokeRect(x, y, w, h);
                ctx.shadowBlur = 0;

                // Draw select face landmark dots
                ctx.fillStyle = isCurrentlyDrowsy ? '#ef4444' : (distracted ? '#fbbf24' : '#60a5fa');
                const dotsToDraw = [...LEFT_EYE_INDICES, ...RIGHT_EYE_INDICES, 1, 33, 263];
                dotsToDraw.forEach(idx => {
                  const lm = landmarks[idx];
                  ctx.beginPath();
                  ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 2.5, 0, 2 * Math.PI);
                  ctx.fill();
                });

                // Labels
                ctx.font = 'bold 11px monospace';
                ctx.fillText('FACE_01', x + 6, y + 18);
                ctx.fillText(isCurrentlyDrowsy ? 'FATIGUE DETECTED' : (distracted ? 'DISTRACTED' : 'SCAN OK'), x + w - 120, y + 18);
              } else {
                faceMissingCount.current++;
                // If face is completely missing, do not raise gaze distraction alert, but draw HUD error
                distracted = false;
                
                ctx.fillStyle = '#ef4444';
                ctx.font = 'bold 14px monospace';
                ctx.fillText('NO FACE DETECTED', canvas.width / 2 - 60, canvas.height / 2);
              }

              // Propagate telemetry updates to Parent state
              onBrowserTelemetryUpdate({
                ear: currentEAR,
                driver_distracted: distracted,
                talking_detected: talkingFlag,
                emotion: currentEAR < 0.20 ? 'sad' : (distracted ? 'surprise' : 'neutral')
              });
            }
          }

          animationFrameId = requestAnimationFrame(processFrame);
        };

        processFrame();
      } catch (err) {
        console.error("Camera Init Error: ", err);
        setCameraStatus('Blocked/Failed');
        setCameraError(true);
      }
    };

    startWebcam();

    return () => {
      active = false;
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [feedMode, talkingFlag]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
      
      {/* Live Video Feed - Left Column (takes 2 spaces on desktop) */}
      <div className="lg:col-span-2 glass-card rounded-2xl p-4 flex flex-col justify-between overflow-hidden relative min-h-[350px]">
        {/* Top bar indicators */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3 z-10">
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" />
            <span className="text-xs font-mono font-bold tracking-wider text-slate-300">
              {feedMode === 'browser' 
                ? `BROWSER WEBCAM FEED // COGNITIVE SCAN (${cameraStatus.toUpperCase()})`
                : (streamError ? 'OFFLINE SIMULATION // COGNITIVE SCAN' : 'LIVE VEHICLE STREAM // COGNITIVE SCAN')}
            </span>
          </div>
          <div className="flex items-center gap-3 text-[10px] font-mono text-slate-400">
            <span>FPS: <strong className="text-emerald-400">29.8</strong></span>
            <span>RES: <strong>1280x720</strong></span>
            <span>SYS_TEMP: <strong>44°C</strong></span>
          </div>
        </div>

        {/* Camera Visualizer Screen */}
        <div className="relative w-full aspect-video rounded-xl bg-slate-950 overflow-hidden flex items-center justify-center border border-slate-900">
          
          {feedMode === 'browser' ? (
            /* Browser Mode - Local Webcam HTML Video Element & Overlay Canvas */
            <>
              <video 
                ref={videoRef}
                playsInline
                muted
                autoPlay
                className="absolute inset-0 w-full h-full object-contain bg-black transform -scale-x-100" 
              />
              <canvas 
                ref={canvasRef}
                className="absolute inset-0 w-full h-full object-contain pointer-events-none transform -scale-x-100"
              />
              
              {/* Fallback display if error or blocked */}
              {cameraError && (
                <div className="absolute inset-0 flex flex-col items-center justify-center p-4 bg-slate-900/90 text-center gap-3">
                  <Camera className="w-12 h-12 text-red-500 animate-bounce" />
                  <p className="text-sm font-semibold text-white">Camera Access Denied or Unavailable</p>
                  <p className="text-xs text-slate-400 max-w-sm">Please click the camera lock icon in your browser URL address bar to grant webcam access, then reload the page.</p>
                </div>
              )}
            </>
          ) : (
            /* Backend Mode - MJPEG stream from python local server */
            <>
              {!streamError ? (
                <img 
                  src={videoFeedUrl} 
                  alt="Live Driver Stream"
                  className="absolute inset-0 w-full h-full object-contain bg-black"
                  onError={() => setStreamError(true)}
                />
              ) : (
                /* Fallback Mock Camera Icon if offline */
                <Camera className="w-12 h-12 text-slate-800 opacity-20 pointer-events-none" />
              )}
            </>
          )}

          {/* Static Scanline Overlay - visible in offline/mock mode */}
          {feedMode === 'backend' && streamError && (
            <>
              <div className="absolute inset-0 bg-linear-to-b from-transparent via-blue-500/2 to-transparent pointer-events-none" />
              <div className="absolute left-0 right-0 h-[2px] bg-blue-500/30 shadow-lg animate-scan pointer-events-none" />
              <div className="absolute inset-0 bg-[radial-gradient(#1e293b_1px,transparent_1px)] [background-size:16px_16px] opacity-40" />
            </>
          )}

          {/* AI Face Bounding Box Mock - only visible in Edge server offline mode */}
          {feedMode === 'backend' && streamError && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className={`w-48 h-56 border-2 rounded-3xl flex flex-col justify-between p-3 transition-all duration-300 ${isDrowsy ? 'border-red-500 shadow-[0_0_30px_rgba(239,68,68,0.2)]' : driver_distracted ? 'border-amber-400 shadow-[0_0_30px_rgba(234,179,8,0.2)]' : 'border-blue-500 shadow-[0_0_30px_rgba(59,130,246,0.15)]'}`}>
                <div className="flex justify-between items-start">
                  <span className="text-[9px] font-mono bg-blue-500/20 text-blue-300 px-1.5 py-0.5 rounded border border-blue-500/30 uppercase leading-none">
                    Face_01
                  </span>
                  <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded border leading-none uppercase ${isDrowsy ? 'bg-red-500/20 text-red-300 border-red-500/30' : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'}`}>
                    {isDrowsy ? 'Fatigue' : 'Active'}
                  </span>
                </div>

                <div className="flex justify-around items-center my-auto w-full px-4">
                  <div className={`w-6 h-6 border-2 border-dashed rounded-full flex items-center justify-center ${isDrowsy ? 'border-red-400 animate-pulse' : 'border-blue-400'}`}>
                    <div className={`w-1.5 h-1.5 rounded-full ${isDrowsy ? 'bg-red-500' : 'bg-blue-400'}`} />
                  </div>
                  <div className={`w-6 h-6 border-2 border-dashed rounded-full flex items-center justify-center ${isDrowsy ? 'border-red-400 animate-pulse' : 'border-blue-400'}`}>
                    <div className={`w-1.5 h-1.5 rounded-full ${isDrowsy ? 'bg-red-500' : 'bg-blue-400'}`} />
                  </div>
                </div>

                <div className="flex justify-between items-end text-[9px] font-mono text-slate-400 leading-none">
                  <span>EAR: {(ear).toFixed(2)}</span>
                  <span>MOOD: {emotion.toUpperCase()}</span>
                </div>
              </div>

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
          <span>Camera Feed Source: {feedMode === 'browser' ? 'Local Browser WebCam' : 'Edge Server (/video-feed)'}</span>
        </div>
      </div>

      {/* Sensor Feeds Details - Right Column */}
      <div className="flex flex-col gap-4">
        
        {/* Interactive Demo Simulation Controls - Displayed when in Browser Webcam Mode */}
        {feedMode === 'browser' && (
          <div className="glass-card p-4 rounded-2xl border border-blue-500/20 bg-[#0f172a]/60 flex flex-col gap-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2 mb-1">
              <h4 className="text-xs font-mono font-bold text-blue-400 flex items-center gap-1.5 uppercase">
                <Sliders className="w-3.5 h-3.5" />
                Demo Simulator Panel
              </h4>
              <span className="text-[9px] font-mono bg-blue-500/10 text-blue-400 px-2 py-0.5 rounded-full border border-blue-500/20 uppercase font-semibold">
                Interactive
              </span>
            </div>

            {/* Phone Toggle */}
            <div className="flex items-center justify-between text-xs text-slate-300">
              <span className="font-semibold">Simulate Phone Use</span>
              <button
                onClick={() => setDemoControls(prev => ({ ...prev, phone_detected: !prev.phone_detected }))}
                className={`px-3 py-1.5 rounded-lg border font-bold text-[10px] tracking-wide transition-all duration-200 cursor-pointer ${demoControls.phone_detected ? 'bg-red-500/25 border-red-500/50 text-red-400 shadow-[0_0_10px_rgba(239,68,68,0.2)]' : 'bg-[#1e293b] border-slate-800 text-slate-400 hover:text-slate-200'}`}
              >
                {demoControls.phone_detected ? 'PHONE ON' : 'PHONE OFF'}
              </button>
            </div>

            {/* Passenger Toggle */}
            <div className="flex items-center justify-between text-xs text-slate-300">
              <span className="font-semibold">Simulate Passenger</span>
              <button
                onClick={() => setDemoControls(prev => ({ ...prev, passenger_detected: !prev.passenger_detected }))}
                className={`px-3 py-1.5 rounded-lg border font-bold text-[10px] tracking-wide transition-all duration-200 cursor-pointer ${demoControls.passenger_detected ? 'bg-blue-500/25 border-blue-500/50 text-blue-400 shadow-[0_0_10px_rgba(59,130,246,0.2)]' : 'bg-[#1e293b] border-slate-800 text-slate-400 hover:text-slate-200'}`}
              >
                {demoControls.passenger_detected ? 'PRESENT' : 'VACANT'}
              </button>
            </div>

            {/* Speed Slider */}
            <div className="flex flex-col gap-1 text-xs text-slate-300 mt-0.5">
              <div className="flex justify-between font-mono">
                <span className="font-semibold">Simulate Driving Speed</span>
                <span className={`font-bold ${demoControls.speed > 80 ? 'text-amber-400 font-extrabold animate-pulse' : 'text-slate-400'}`}>
                  {demoControls.speed} km/h
                </span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="120" 
                value={demoControls.speed}
                onChange={(e) => setDemoControls(prev => ({ ...prev, speed: parseInt(e.target.value) }))}
                className="w-full accent-blue-500 h-1 bg-slate-800 rounded-lg cursor-pointer mt-1"
              />
              {demoControls.speed > 80 && (
                <span className="text-[9px] font-mono text-amber-400 mt-1 uppercase">
                  ⚠️ Speed &gt; 80km/h amplifies cognitive fatigue warnings
                </span>
              )}
            </div>

            {/* Device Diagnostics */}
            <div className="grid grid-cols-2 gap-2 text-[9px] font-mono border-t border-slate-850 pt-2.5 mt-1.5 text-slate-400">
              <div className="flex justify-between">
                <span>WEBCAM:</span>
                <span className={cameraStatus === 'Active' ? 'text-emerald-400 font-bold' : 'text-amber-500'}>
                  {cameraStatus.toUpperCase()}
                </span>
              </div>
              <div className="flex justify-between">
                <span>MICROPHONE:</span>
                <span className={micStatus === 'Active' ? 'text-emerald-400 font-bold' : 'text-amber-500'}>
                  {micStatus.toUpperCase()}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Phone Detection Status Card */}
        <div 
          onClick={() => {
            if (feedMode === 'browser') {
              setDemoControls(prev => ({ ...prev, phone_detected: !prev.phone_detected }));
            }
          }}
          className={`glass-card p-4 rounded-2xl flex items-center justify-between border transition-all duration-250 ${feedMode === 'browser' ? 'cursor-pointer hover:bg-slate-800/30 hover:border-blue-500/20 active:scale-[0.98]' : ''} ${phone_detected ? 'border-red-500/30 bg-red-500/5' : 'border-transparent'}`}
        >
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-xl border ${phone_detected ? 'bg-red-500/20 border-red-500/30 text-red-400 animate-bounce' : 'bg-slate-800/50 border-slate-700/50 text-slate-400'}`}>
              {phone_detected ? <PhoneCall className="w-5 h-5" /> : <PhoneOff className="w-5 h-5" />}
            </div>
            <div>
              <h4 className="text-sm font-semibold text-white flex items-center gap-1.5">
                Phone Detection
                {feedMode === 'browser' && <span className="text-[9px] text-blue-400 font-mono">(Simulate)</span>}
              </h4>
              <p className="text-xs text-slate-400 mt-0.5">YOLOv8 Class 67 Monitor</p>
            </div>
          </div>
          <span className={`text-xs font-mono font-bold px-2 py-1 rounded-lg uppercase ${phone_detected ? 'bg-red-500/20 text-red-400 text-neon-red' : 'bg-slate-800 text-slate-400'}`}>
            {phone_detected ? 'Detected' : 'Clear'}
          </span>
        </div>

        {/* Passenger Detection Status Card */}
        <div 
          onClick={() => {
            if (feedMode === 'browser') {
              setDemoControls(prev => ({ ...prev, passenger_detected: !prev.passenger_detected }));
            }
          }}
          className={`glass-card p-4 rounded-2xl flex items-center justify-between border transition-all duration-250 ${feedMode === 'browser' ? 'cursor-pointer hover:bg-slate-800/30 hover:border-blue-500/20 active:scale-[0.98]' : ''} ${passenger_detected ? 'border-blue-500/20 bg-blue-500/5' : 'border-transparent'}`}
        >
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-xl border ${passenger_detected ? 'bg-blue-500/20 border-blue-500/30 text-blue-400' : 'bg-slate-800/50 border-slate-700/50 text-slate-400'}`}>
              {passenger_detected ? <Users className="w-5 h-5" /> : <Users2 className="w-5 h-5" />}
            </div>
            <div>
              <h4 className="text-sm font-semibold text-white flex items-center gap-1.5">
                Passenger Presence
                {feedMode === 'browser' && <span className="text-[9px] text-blue-400 font-mono">(Simulate)</span>}
              </h4>
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
