import React, { useState, useEffect } from 'react';
import { Shield, Cpu, RefreshCw, LogOut } from 'lucide-react';

export default function Header({ isOnline, lastSync }) {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formattedTime = time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const formattedDate = time.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });

  return (
    <header className="glass-card px-6 py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-4 rounded-2xl mb-6">
      {/* Brand Logo & Name */}
      <div className="flex items-center gap-3">
        <div className="bg-blue-600/20 p-2.5 rounded-xl border border-blue-500/30 flex items-center justify-center animate-pulse-slow">
          <Shield className="w-6 h-6 text-blue-400" />
        </div>
        <div>
          <h1 className="text-xl font-bold tracking-wider text-white flex items-center gap-2">
            DRIVE<span className="text-blue-500">WISE</span>
            <span className="text-[10px] bg-blue-500/10 text-blue-400 px-2 py-0.5 rounded-full border border-blue-500/20 uppercase font-mono">
              AI Copilot
            </span>
          </h1>
          <p className="text-xs text-slate-400">Driver Safety Monitoring Hub</p>
        </div>
      </div>

      {/* System Status Metrics */}
      <div className="flex flex-wrap items-center gap-4 md:gap-6 text-sm">
        {/* Server Sync State */}
        <div className="flex items-center gap-2">
          <div className="relative flex h-2.5 w-2.5">
            {isOnline && (
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            )}
            <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${isOnline ? 'bg-emerald-500' : 'bg-red-500'}`}></span>
          </div>
          <span className="text-slate-300 font-medium">
            {isOnline ? 'System Online (Direct)' : 'Local Simulation'}
          </span>
        </div>

        {/* Live Clock */}
        <div className="border-l border-slate-800 pl-4 md:pl-6 text-left">
          <p className="text-white font-mono text-base font-semibold leading-none">{formattedTime}</p>
          <p className="text-[10px] text-slate-400 font-medium mt-1 uppercase">{formattedDate}</p>
        </div>

        {/* Driver Profile */}
        <div className="border-l border-slate-800 pl-4 md:pl-6 flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center font-bold text-blue-400 shadow-inner">
            JD
          </div>
          <div className="text-left">
            <h4 className="text-sm font-semibold text-white leading-none">John Doe</h4>
            <p className="text-[10px] text-slate-400 font-medium mt-1">Route ID: TX-882</p>
          </div>
        </div>
      </div>
    </header>
  );
}
