import React from 'react';
import EmotionPieChart from '../charts/EmotionPieChart';
import EventsBarChart from '../charts/EventsBarChart';
import TalkingDuration from '../charts/TalkingDuration';
import { Heart, Activity, Volume2 } from 'lucide-react';

export default function AttentionAnalytics({ emotionStats, eventHistory, talkingHistory }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
      
      {/* 1. Emotion Distribution Chart */}
      <div className="glass-card rounded-2xl p-5 flex flex-col justify-between">
        <div>
          <h3 className="text-sm font-bold text-white tracking-wide uppercase flex items-center gap-2">
            <Heart className="w-4 h-4 text-blue-400" />
            Emotion Breakdown
          </h3>
          <p className="text-[11px] text-slate-400 mt-1">Facial layout distribution profile</p>
        </div>
        <div className="flex-1 mt-4">
          <EmotionPieChart emotionStats={emotionStats} />
        </div>
      </div>

      {/* 2. Drowsiness & Distraction Events */}
      <div className="glass-card rounded-2xl p-5 flex flex-col justify-between">
        <div>
          <h3 className="text-sm font-bold text-white tracking-wide uppercase flex items-center gap-2">
            <Activity className="w-4 h-4 text-blue-400" />
            Cognitive Anomalies
          </h3>
          <p className="text-[11px] text-slate-400 mt-1">Drowsiness vs distraction event trends</p>
        </div>
        <div className="flex-1 mt-4">
          <EventsBarChart eventHistory={eventHistory} />
        </div>
      </div>

      {/* 3. Talking Duration / Sound Levels */}
      <div className="glass-card rounded-2xl p-5 flex flex-col justify-between">
        <div>
          <h3 className="text-sm font-bold text-white tracking-wide uppercase flex items-center gap-2">
            <Volume2 className="w-4 h-4 text-blue-400" />
            Acoustic Activity Log
          </h3>
          <p className="text-[11px] text-slate-400 mt-1">Microphone audio volume levels (dB)</p>
        </div>
        <div className="flex-1 mt-4">
          <TalkingDuration talkingHistory={talkingHistory} />
        </div>
      </div>

    </div>
  );
}
