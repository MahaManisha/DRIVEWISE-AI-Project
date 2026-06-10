import React from 'react';
import { Cpu } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="glass-card mt-6 px-6 py-4 flex flex-col sm:flex-row items-center justify-between text-slate-500 rounded-2xl gap-3 text-xs font-semibold">
      <div className="flex items-center gap-2">
        <Cpu className="w-4 h-4 text-slate-600" />
        <span>DriveWise AI Driver Safety & Monitoring System</span>
      </div>
      <div className="flex items-center gap-4">
        <span>© {new Date().getFullYear()} DriveWise Tech. All rights reserved.</span>
        <span className="bg-slate-800 text-slate-400 border border-slate-700 px-2 py-0.5 rounded-md font-mono">
          V1.0.0
        </span>
      </div>
    </footer>
  );
}
