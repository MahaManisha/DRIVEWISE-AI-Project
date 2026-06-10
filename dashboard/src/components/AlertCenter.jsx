import React from 'react';
import { BellRing, ShieldAlert, CheckCircle, VolumeX } from 'lucide-react';

export default function AlertCenter({ alerts, onAcknowledge, onMute }) {
  // Severity Badge color selector
  const getSeverityBadge = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'high':
        return 'bg-red-500/20 text-red-400 border-red-500/30';
      case 'medium':
        return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
      default:
        return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    }
  };

  return (
    <div className="glass-card rounded-2xl p-5 mb-6">
      <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-4">
        <div>
          <h3 className="text-sm font-bold text-white tracking-wide uppercase flex items-center gap-2">
            <BellRing className="w-4.5 h-4.5 text-red-400 animate-pulse" />
            Live Alert Center
          </h3>
          <p className="text-[11px] text-slate-400 mt-1">Real-time cabin incidents log</p>
        </div>
        
        {/* Quick controls */}
        <button 
          onClick={onMute}
          className="text-xs flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium transition cursor-pointer"
        >
          <VolumeX className="w-3.5 h-3.5" />
          Mute Alarms
        </button>
      </div>

      {/* Responsive Table Container */}
      <div className="overflow-x-auto max-h-[220px] overflow-y-auto">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 uppercase font-mono tracking-wider font-semibold">
              <th className="py-2.5 px-4">Timestamp</th>
              <th className="py-2.5 px-4">Alert Type</th>
              <th className="py-2.5 px-4">Severity</th>
              <th className="py-2.5 px-4">Status</th>
              <th className="py-2.5 px-4 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-900 font-medium">
            {alerts.length === 0 ? (
              <tr>
                <td colSpan={5} className="py-8 text-center text-slate-500 font-mono">
                  No active safety alerts. Driving conditions optimal.
                </td>
              </tr>
            ) : (
              alerts.map((alert, idx) => (
                <tr key={`${alert.time}-${idx}`} className="hover:bg-slate-800/20 transition-all duration-200">
                  <td className="py-3 px-4 text-slate-300 font-mono">{alert.time}</td>
                  <td className="py-3 px-4 font-bold text-white flex items-center gap-1.5">
                    <ShieldAlert className={`w-3.5 h-3.5 ${alert.severity.toLowerCase() === 'high' ? 'text-red-400' : 'text-amber-400'}`} />
                    {alert.type}
                  </td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-0.5 rounded-md border text-[10px] uppercase font-mono font-bold ${getSeverityBadge(alert.severity)}`}>
                      {alert.severity}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-slate-400 flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-ping" />
                    <span>Active Trigger</span>
                  </td>
                  <td className="py-3 px-4 text-right">
                    <button 
                      onClick={() => onAcknowledge(idx)}
                      className="text-[10px] inline-flex items-center gap-1 px-2.5 py-1 rounded-md border border-emerald-500/30 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 font-bold transition cursor-pointer"
                    >
                      <CheckCircle className="w-3 h-3" />
                      Resolve
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
