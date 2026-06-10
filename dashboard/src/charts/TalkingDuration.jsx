import React from 'react';
import { 
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer 
} from 'recharts';

export default function TalkingDuration({ talkingHistory }) {
  // Safe default mockup dataset representing speech activity level (decibels/volume) over time
  const defaultHistory = [
    { time: '14:20', level: 12 },
    { time: '14:21', level: 45 },
    { time: '14:22', level: 18 },
    { time: '14:23', level: 60 },
    { time: '14:24', level: 75 },
    { time: '14:25', level: 15 },
    { time: '14:26', level: 10 },
    { time: '14:27', level: 90 },
    { time: '14:28', level: 20 },
    { time: '14:29', level: 5 }
  ];

  const data = talkingHistory && talkingHistory.length > 0 ? talkingHistory : defaultHistory;

  return (
    <div className="w-full h-full min-h-[200px]">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={data}
          margin={{ top: 10, right: 10, left: -25, bottom: 0 }}
        >
          <defs>
            <linearGradient id="talkingGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.4}/>
              <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <XAxis 
            dataKey="time" 
            stroke="#475569" 
            fontSize={9}
            tickLine={false}
            dy={8}
          />
          <YAxis 
            stroke="#475569" 
            fontSize={9}
            tickLine={false}
            dx={-8}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#0f172a', 
              borderColor: 'rgba(255, 255, 255, 0.1)',
              borderRadius: '12px',
              fontSize: '11px'
            }}
          />
          <Area 
            type="monotone" 
            dataKey="level" 
            stroke="#f59e0b" 
            strokeWidth={2}
            fillOpacity={1} 
            fill="url(#talkingGradient)" 
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
