import React from 'react';
import RiskGauge from '../charts/RiskGauge';
import RiskTrend from '../charts/RiskTrend';
import { AlertCircle, TrendingUp } from 'lucide-react';

export default function RiskAnalysis({ currentScore, level, history }) {
  // Config matching risk status
  const getLevelConfig = (lvl) => {
    switch (lvl) {
      case 'HIGH RISK':
        return {
          text: 'HIGH RISK LEVEL DETECTED',
          desc: 'Telemetry reports active warning triggers. Immediate corrective action advised.',
          badge: 'bg-red-500/20 text-red-400 border-red-500/30'
        };
      case 'WARNING':
        return {
          text: 'WARNING STATUS ENCOUNTERED',
          desc: 'High drowsiness index or distracted head posture. Monitor conditions.',
          badge: 'bg-amber-500/20 text-amber-400 border-amber-500/30'
        };
      default:
        return {
          text: 'DRIVING CONDITIONS NORMAL',
          desc: 'All cognitive metrics register inside safe baseline bounds.',
          badge: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
        };
    }
  };

  const config = getLevelConfig(level);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
      
      {/* 1. Risk Score Gauge Card */}
      <div className="glass-card rounded-2xl p-5 flex flex-col justify-between">
        <div>
          <h3 className="text-sm font-bold text-white tracking-wide uppercase flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-blue-400" />
            Risk Gauge Index
          </h3>
          <p className="text-[11px] text-slate-400 mt-1">Unified safety score compilation</p>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <RiskGauge score={currentScore} />
        </div>
        <div className={`p-3 rounded-xl border ${config.badge} text-center`}>
          <h4 className="text-xs font-black tracking-wider uppercase leading-none mb-1">
            {config.text}
          </h4>
          <p className="text-[9px] opacity-80 leading-normal">
            {config.desc}
          </p>
        </div>
      </div>

      {/* 2. Risk Trend Line Chart (Takes 2 Columns on Desktop) */}
      <div className="lg:col-span-2 glass-card rounded-2xl p-5 flex flex-col justify-between">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-bold text-white tracking-wide uppercase flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-blue-400" />
              Risk Timeline Trend
            </h3>
            <p className="text-[11px] text-slate-400 mt-1">Real-time danger score fluctuation chart</p>
          </div>
          <span className="text-[10px] bg-slate-800 text-slate-400 font-mono px-2 py-1 rounded-md border border-slate-700">
            INTERVAL: 2s
          </span>
        </div>
        <div className="flex-1 min-h-[200px]">
          <RiskTrend history={history} />
        </div>
      </div>

    </div>
  );
}
