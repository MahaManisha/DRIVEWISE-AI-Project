import React from 'react';

export default function RiskGauge({ score }) {
  // Normalize score from 0 - 20
  const normalizedScore = Math.max(0, Math.min(20, score));
  
  // Circumference of semi-circle (r=50, C=2*pi*r, half is pi*r ~ 157)
  const radius = 50;
  const strokeWidth = 10;
  const strokeDasharray = radius * Math.PI; // ~157.08
  const progressPercent = normalizedScore / 20;
  const strokeDashoffset = strokeDasharray * (1 - progressPercent);

  // Determine color matching risk status
  const getColor = (val) => {
    if (val >= 11) return '#f87171'; // Red
    if (val >= 6) return '#fbbf24';  // Orange/Yellow
    return '#34d399';                // Emerald
  };

  const getGlowColor = (val) => {
    if (val >= 11) return 'rgba(239, 68, 68, 0.4)';
    if (val >= 6) return 'rgba(234, 179, 8, 0.4)';
    return 'rgba(52, 211, 153, 0.4)';
  };

  const activeColor = getColor(normalizedScore);
  const glowColor = getGlowColor(normalizedScore);

  return (
    <div className="flex flex-col items-center justify-center p-4 h-full relative">
      <div className="w-44 h-28 relative">
        <svg 
          viewBox="0 0 120 70" 
          className="w-full h-full"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Background Arc */}
          <path
            d="M 10,60 A 50,50 0 0,1 110,60"
            fill="none"
            stroke="#1e293b"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
          {/* Colored Progress Arc */}
          <path
            d="M 10,60 A 50,50 0 0,1 110,60"
            fill="none"
            stroke={activeColor}
            strokeWidth={strokeWidth}
            strokeDasharray={strokeDasharray}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            style={{ 
              transition: 'stroke-dashoffset 0.8s ease-in-out, stroke 0.5s ease',
              filter: `drop-shadow(0px 0px 4px ${glowColor})`
            }}
          />
        </svg>

        {/* Floating Text readouts */}
        <div className="absolute inset-0 flex flex-col items-center justify-end pb-1 text-center">
          <span className="text-3xl font-black font-mono leading-none tracking-tight text-white">
            {normalizedScore}
          </span>
          <span className="text-[10px] text-slate-500 font-bold uppercase mt-1">
            Max: 20
          </span>
        </div>
      </div>

      {/* Threshold Legends */}
      <div className="flex justify-between w-full text-[9px] font-mono text-slate-500 mt-4 px-2">
        <span className="text-emerald-400">SAFE (0-5)</span>
        <span className="text-amber-400">WARNING (6-10)</span>
        <span className="text-red-400">CRITICAL (11+)</span>
      </div>
    </div>
  );
}
