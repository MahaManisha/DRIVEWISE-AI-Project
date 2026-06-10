import React from 'react';
import { 
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer 
} from 'recharts';

export default function RiskTrend({ history }) {
  // Safe mapping of historical data
  const data = history.map(item => ({
    time: item.time,
    score: item.risk_score
  }));

  return (
    <div className="w-full h-full min-h-[180px]">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={data}
          margin={{ top: 10, right: 10, left: -25, bottom: 0 }}
        >
          <defs>
            <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4}/>
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
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
            domain={[0, 20]} 
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
              fontSize: '11px',
              color: '#fff'
            }}
            labelClassName="font-mono text-slate-400 font-semibold"
          />
          <Area 
            type="monotone" 
            dataKey="score" 
            stroke="#3b82f6" 
            strokeWidth={2}
            fillOpacity={1} 
            fill="url(#riskGradient)" 
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
