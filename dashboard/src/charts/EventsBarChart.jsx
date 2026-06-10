import React from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';

export default function EventsBarChart({ eventHistory }) {
  // Pre-formatted hourly data list representing drowsiness & distraction events
  const defaultHistory = [
    { hour: '10:00', Drowsiness: 1, Distraction: 2 },
    { hour: '11:00', Drowsiness: 0, Distraction: 4 },
    { hour: '12:00', Drowsiness: 2, Distraction: 1 },
    { hour: '13:00', Drowsiness: 3, Distraction: 5 },
    { hour: '14:00', Drowsiness: 1, Distraction: 2 },
    { hour: '15:00', Drowsiness: 4, Distraction: 3 }
  ];

  const data = eventHistory && eventHistory.length > 0 ? eventHistory : defaultHistory;

  return (
    <div className="w-full h-full min-h-[200px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 10, right: 10, left: -25, bottom: 0 }}
        >
          <XAxis 
            dataKey="hour" 
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
          <Legend 
            verticalAlign="top" 
            height={36} 
            iconType="square"
            iconSize={8}
            wrapperStyle={{ fontSize: '10px' }}
          />
          <Bar 
            dataKey="Drowsiness" 
            fill="#f87171" 
            radius={[4, 4, 0, 0]} 
          />
          <Bar 
            dataKey="Distraction" 
            fill="#fbbf24" 
            radius={[4, 4, 0, 0]} 
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
