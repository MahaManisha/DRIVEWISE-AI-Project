import React from 'react';
import { 
  Award, Clock, Eye, Phone, AlertOctagon 
} from 'lucide-react';

export default function SafetySummary({ summary }) {
  const {
    safety_score = 98,
    drive_time = "3h 45m",
    drowsy_count = 0,
    phone_count = 0,
    distraction_count = 0
  } = summary;

  // Progress circle SVG calculation
  const radius = 38;
  const stroke = 6;
  const normalizedRadius = radius - stroke * 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (safety_score / 100) * circumference;

  const getScoreColor = (score) => {
    if (score >= 90) return 'text-emerald-400 stroke-emerald-400';
    if (score >= 75) return 'text-amber-400 stroke-amber-400';
    return 'text-red-400 stroke-red-400';
  };

  const scoreColor = getScoreColor(safety_score);

  return (
    <div className="glass-card rounded-2xl p-5 mb-6">
      <h3 className="text-sm font-bold text-white tracking-wide uppercase flex items-center gap-2 border-b border-slate-800 pb-4 mb-4">
        <Award className="w-4.5 h-4.5 text-blue-400" />
        Trip Safety Summary
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 items-center">
        
        {/* 1. Circular Progress Gauge (Safety Score %) */}
        <div className="flex items-center gap-4 border-r border-slate-800/50 pr-4">
          <div className="relative w-20 h-20 flex items-center justify-center">
            <svg className="w-full h-full transform -rotate-90">
              <circle
                className="stroke-slate-800 fill-transparent"
                strokeWidth={stroke}
                r={normalizedRadius}
                cx={radius + 2}
                cy={radius + 2}
              />
              <circle
                className={`fill-transparent transition-all duration-700 ${scoreColor}`}
                strokeWidth={stroke}
                strokeDasharray={circumference + ' ' + circumference}
                style={{ strokeDashoffset }}
                strokeLinecap="round"
                r={normalizedRadius}
                cx={radius + 2}
                cy={radius + 2}
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center text-center">
              <span className="text-lg font-black font-mono text-white leading-none">
                {safety_score}%
              </span>
            </div>
          </div>
          <div>
            <h4 className="text-xs font-bold text-slate-300">Driver Safety Index</h4>
            <p className="text-[10px] text-slate-500 font-medium mt-0.5">Calculated trip safety average</p>
          </div>
        </div>

        {/* 2. Total Driving Time */}
        <div className="flex items-center gap-3 border-r border-slate-800/50 pr-4">
          <div className="bg-slate-800/50 border border-slate-700/50 p-2 rounded-xl text-blue-400">
            <Clock className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-slate-400 text-[10px] uppercase font-mono tracking-wider font-semibold">Total Drive Time</h4>
            <p className="text-lg font-black font-mono text-white leading-tight mt-0.5">{drive_time}</p>
          </div>
        </div>

        {/* 3. Drowsiness Events Counter */}
        <div className="flex items-center gap-3 border-r border-slate-800/50 pr-4">
          <div className="bg-slate-800/50 border border-slate-700/50 p-2 rounded-xl text-red-400">
            <Eye className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-slate-400 text-[10px] uppercase font-mono tracking-wider font-semibold">Drowsiness Triggers</h4>
            <p className="text-lg font-black font-mono text-white leading-tight mt-0.5">{drowsy_count}</p>
          </div>
        </div>

        {/* 4. Phone Usage Counter */}
        <div className="flex items-center gap-3 border-r border-slate-800/50 pr-4">
          <div className="bg-slate-800/50 border border-slate-700/50 p-2 rounded-xl text-red-500">
            <Phone className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-slate-400 text-[10px] uppercase font-mono tracking-wider font-semibold">Phone Usage Incidents</h4>
            <p className="text-lg font-black font-mono text-white leading-tight mt-0.5">{phone_count}</p>
          </div>
        </div>

        {/* 5. Distraction Events Counter */}
        <div className="flex items-center gap-3">
          <div className="bg-slate-800/50 border border-slate-700/50 p-2 rounded-xl text-amber-500">
            <AlertOctagon className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-slate-400 text-[10px] uppercase font-mono tracking-wider font-semibold">Distraction Incidents</h4>
            <p className="text-lg font-black font-mono text-white leading-tight mt-0.5">{distraction_count}</p>
          </div>
        </div>

      </div>
    </div>
  );
}
