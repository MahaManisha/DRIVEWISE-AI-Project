import React from 'react';
import { 
  AlertTriangle, Shield, Gauge, Eye, 
  Smile, Frown, Meh, Skull, Activity, Bell 
} from 'lucide-react';

export default function KPICards({ data }) {
  const {
    risk_score = 0,
    risk_level = "SAFE",
    speed = 0,
    ear = 0.28,
    emotion = "neutral",
    alerts_today = 0
  } = data;

  // Emotion configuration helper
  const getEmotionDetails = (emo) => {
    const map = {
      neutral: { label: 'Neutral', color: 'text-slate-400 bg-slate-500/10 border-slate-500/20', icon: Meh },
      happy: { label: 'Focused / Calm', color: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20', icon: Smile },
      sad: { label: 'Fatigued / Sad', color: 'text-amber-400 bg-amber-500/10 border-amber-500/20', icon: Frown },
      fear: { label: 'Panic / Stressed', color: 'text-rose-400 bg-rose-500/10 border-rose-500/20', icon: Skull },
      anger: { label: 'Aggressive / Angry', color: 'text-red-400 bg-red-500/10 border-red-500/20', icon: FlameIcon },
      surprise: { label: 'Distracted', color: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20', icon: Activity }
    };
    return map[emo?.toLowerCase()] || { label: emo || 'Unknown', color: 'text-slate-400 bg-slate-500/10', icon: Meh };
  };

  function FlameIcon(props) {
    return (
      <svg
        {...props}
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="lucide lucide-flame w-5 h-5 text-red-400"
      >
        <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z" />
      </svg>
    );
  }

  // Risk Level styles helper
  const getRiskStyles = (level) => {
    switch(level) {
      case 'HIGH RISK':
        return {
          bg: 'bg-red-500/10 border-red-500/30 text-red-400',
          indicator: 'bg-red-500 bg-neon-red text-neon-red',
          text: 'text-red-400'
        };
      case 'WARNING':
        return {
          bg: 'bg-amber-500/10 border-amber-500/30 text-amber-400',
          indicator: 'bg-amber-500 bg-neon-yellow text-neon-yellow',
          text: 'text-amber-400'
        };
      default:
        return {
          bg: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400',
          indicator: 'bg-emerald-500 bg-neon-green text-neon-green',
          text: 'text-emerald-400'
        };
    }
  };

  const riskStyles = getRiskStyles(risk_level);
  const emoDetails = getEmotionDetails(emotion);
  const EmoIcon = emoDetails.icon;

  const isDrowsyEAR = ear < 0.20;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4 mb-6">
      
      {/* 1. Risk Score Card */}
      <div className="glass-card glass-card-hover p-4 rounded-2xl flex flex-col justify-between min-h-[120px]">
        <div className="flex items-center justify-between text-slate-400 mb-2">
          <span className="text-xs font-semibold tracking-wide uppercase">Risk Score</span>
          <AlertTriangle className={`w-5 h-5 ${risk_score >= 11 ? 'text-red-400' : risk_score >= 6 ? 'text-amber-400' : 'text-blue-400'}`} />
        </div>
        <div>
          <span className={`text-2xl font-black font-mono tracking-tight leading-none ${risk_score >= 11 ? 'text-red-400' : risk_score >= 6 ? 'text-amber-400' : 'text-slate-200'}`}>
            {risk_score}
          </span>
          <span className="text-xs text-slate-500 ml-1">/ 20</span>
          <div className="w-full bg-slate-800 h-1.5 rounded-full mt-2 overflow-hidden">
            <div 
              className={`h-full rounded-full transition-all duration-500 ${risk_score >= 11 ? 'bg-red-500' : risk_score >= 6 ? 'bg-amber-500' : 'bg-blue-500'}`}
              style={{ width: `${(risk_score / 20) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* 2. Driver Status Card */}
      <div className={`glass-card glass-card-hover p-4 rounded-2xl border ${riskStyles.bg} flex flex-col justify-between min-h-[120px]`}>
        <div className="flex items-center justify-between text-slate-400 mb-2">
          <span className="text-xs font-semibold tracking-wide uppercase">Driver Status</span>
          <Shield className="w-5 h-5" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className={`w-2.5 h-2.5 rounded-full ${riskStyles.indicator}`} />
            <h3 className="text-lg font-black tracking-wide leading-none uppercase">
              {risk_level}
            </h3>
          </div>
          <p className="text-[10px] text-slate-400 mt-2 font-medium">Real-time status analysis</p>
        </div>
      </div>

      {/* 3. Current Speed Card */}
      <div className="glass-card glass-card-hover p-4 rounded-2xl flex flex-col justify-between min-h-[120px]">
        <div className="flex items-center justify-between text-slate-400 mb-2">
          <span className="text-xs font-semibold tracking-wide uppercase">Current Speed</span>
          <Gauge className={`w-5 h-5 ${speed > 100 ? 'text-red-400' : speed > 80 ? 'text-amber-400' : 'text-slate-400'}`} />
        </div>
        <div>
          <span className="text-2xl font-black font-mono text-slate-200 tracking-tight leading-none">
            {speed}
          </span>
          <span className="text-xs text-slate-500 ml-1">km/h</span>
          <p className="text-[10px] text-slate-400 mt-2 font-medium">
            {speed > 80 ? '⚠️ High-speed risk' : 'Normal range'}
          </p>
        </div>
      </div>

      {/* 4. Eye Aspect Ratio (EAR) Card */}
      <div className={`glass-card glass-card-hover p-4 rounded-2xl border ${isDrowsyEAR ? 'border-red-500/30 bg-red-500/5' : 'border-transparent'} flex flex-col justify-between min-h-[120px]`}>
        <div className="flex items-center justify-between text-slate-400 mb-2">
          <span className="text-xs font-semibold tracking-wide uppercase">Eye Aspect (EAR)</span>
          <Eye className={`w-5 h-5 ${isDrowsyEAR ? 'text-red-400 animate-pulse' : 'text-slate-400'}`} />
        </div>
        <div>
          <span className={`text-2xl font-black font-mono tracking-tight leading-none ${isDrowsyEAR ? 'text-red-400' : 'text-slate-200'}`}>
            {ear.toFixed(2)}
          </span>
          <p className="text-[10px] text-slate-400 mt-2 font-medium">
            {isDrowsyEAR ? '🚨 Warning: Eyes Closed!' : 'Normal (Limit: >= 0.20)'}
          </p>
        </div>
      </div>

      {/* 5. Emotion Card */}
      <div className="glass-card glass-card-hover p-4 rounded-2xl flex flex-col justify-between min-h-[120px]">
        <div className="flex items-center justify-between text-slate-400 mb-2">
          <span className="text-xs font-semibold tracking-wide uppercase">Mood / State</span>
          <EmoIcon className="w-5 h-5" />
        </div>
        <div>
          <span className={`inline-block px-2.5 py-1 text-xs rounded-xl font-bold border ${emoDetails.color}`}>
            {emoDetails.label}
          </span>
          <p className="text-[10px] text-slate-400 mt-2 font-medium">Facial layout scoring</p>
        </div>
      </div>

      {/* 6. Alert Count Today Card */}
      <div className="glass-card glass-card-hover p-4 rounded-2xl flex flex-col justify-between min-h-[120px]">
        <div className="flex items-center justify-between text-slate-400 mb-2">
          <span className="text-xs font-semibold tracking-wide uppercase">Alerts Today</span>
          <Bell className="w-5 h-5 text-blue-400" />
        </div>
        <div>
          <span className="text-2xl font-black font-mono text-slate-200 tracking-tight leading-none">
            {alerts_today}
          </span>
          <p className="text-[10px] text-slate-400 mt-2 font-medium">Cumulative events logged</p>
        </div>
      </div>

    </div>
  );
}
