import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { GlassCard } from '../ui/GlassCard';
import { BarChart2 } from 'lucide-react';

interface ConfidenceChartProps {
  probabilities: Record<string, number> | null;
}

export const ConfidenceChart: React.FC<ConfidenceChartProps> = ({ probabilities }) => {
  if (!probabilities) return null;

  const data = Object.entries(probabilities).map(([name, value]) => ({
    name: name.charAt(0).toUpperCase() + name.slice(1),
    value: Number((value * 100).toFixed(1)),
    isCancer: name.toLowerCase().includes('cancer') || name.toLowerCase().includes('tumor')
  }));

  return (
    <GlassCard title="Class Probabilities" icon={<BarChart2 className="w-5 h-5" />} className="h-full">
      <div className="h-[250px] w-full mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={true} vertical={false} />
            <XAxis type="number" domain={[0, 100]} stroke="#94a3b8" tickFormatter={(val) => `${val}%`} />
            <YAxis dataKey="name" type="category" stroke="#f8fafc" width={100} tick={{ fill: '#e2e8f0', fontSize: 12 }} />
            <Tooltip 
              cursor={{ fill: 'rgba(255,255,255,0.05)' }}
              contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
              itemStyle={{ color: '#f8fafc' }}
              formatter={(value: number) => [`${value}%`, 'Probability']}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={32}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.isCancer ? '#ef4444' : '#10b981'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </GlassCard>
  );
};
