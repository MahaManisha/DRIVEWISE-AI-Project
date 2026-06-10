import React from 'react';
import { 
  PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip 
} from 'recharts';

export default function EmotionPieChart({ emotionStats }) {
  // Map raw emotion keys to descriptive labels
  const data = [
    { name: 'Focused (Calm)', value: emotionStats.happy || 0, color: '#10b981' },
    { name: 'Neutral', value: emotionStats.neutral || 0, color: '#64748b' },
    { name: 'Tired (Sad)', value: emotionStats.sad || 0, color: '#f59e0b' },
    { name: 'Stressed (Fear)', value: emotionStats.fear || 0, color: '#ec4899' },
    { name: 'Aggressive (Anger)', value: emotionStats.anger || 0, color: '#ef4444' },
    { name: 'Distracted (Surprise)', value: emotionStats.surprise || 0, color: '#06b6d4' }
  ].filter(item => item.value > 0);

  // Fallback if no data yet
  const displayData = data.length > 0 ? data : [
    { name: 'No Data Yet', value: 1, color: '#1e293b' }
  ];

  return (
    <div className="w-full h-full min-h-[200px] flex items-center justify-center">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={displayData}
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={80}
            paddingAngle={3}
            dataKey="value"
          >
            {displayData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#0f172a', 
              borderColor: 'rgba(255, 255, 255, 0.1)',
              borderRadius: '12px',
              fontSize: '11px'
            }}
          />
          <Legend 
            verticalAlign="bottom" 
            height={36} 
            iconType="circle"
            iconSize={8}
            wrapperStyle={{ fontSize: '10px', color: '#94a3b8' }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
