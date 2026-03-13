import { useState, useMemo } from 'react';

// Generates mock portfolio value over time
function generateEquityCurve(days = 90) {
  const points = [];
  let value = 100000;
  const now = Date.now();
  for (let i = days; i >= 0; i--) {
    value *= 1 + (Math.random() - 0.47) * 0.015;
    points.push({
      date: new Date(now - i * 86400000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      value: Math.round(value),
    });
  }
  return points;
}

const DATA = generateEquityCurve(90);

export default function AnalyticsChart() {
  const [hover, setHover] = useState(null);
  const W = 600, H = 160, PAD = { top: 10, right: 20, bottom: 30, left: 70 };

  const values = DATA.map((d) => d.value);
  const minV = Math.min(...values);
  const maxV = Math.max(...values);
  const range = maxV - minV || 1;

  const pts = DATA.map((d, i) => ({
    x: PAD.left + (i / (DATA.length - 1)) * (W - PAD.left - PAD.right),
    y: PAD.top + (1 - (d.value - minV) / range) * (H - PAD.top - PAD.bottom),
    ...d,
  }));

  const linePath = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ');
  const areaPath = `${linePath} L${pts[pts.length - 1].x},${H - PAD.bottom} L${pts[0].x},${H - PAD.bottom} Z`;

  const isProfit = pts[pts.length - 1].value > pts[0].value;
  const color = isProfit ? '#34d399' : '#f87171';

  const labels = [0, Math.floor(DATA.length / 2), DATA.length - 1];

  const hoverPt = hover != null ? pts[hover] : null;

  const totalReturn = ((pts[pts.length - 1].value - pts[0].value) / pts[0].value * 100).toFixed(2);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-widest text-zinc-500 font-mono">Equity Curve</div>
          <div className="text-xs font-mono text-zinc-400 mt-0.5">90-Day Performance</div>
        </div>
        <div className="text-right">
          <div className={`text-lg font-light tabular-nums ${isProfit ? 'text-emerald-400' : 'text-red-400'}`}>
            {isProfit ? '+' : ''}{totalReturn}%
          </div>
          <div className="text-xs text-zinc-500 font-mono">
            ${pts[pts.length - 1].value.toLocaleString()}
          </div>
        </div>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ height: 160 }}
        onMouseLeave={() => setHover(null)}
      >
        <defs>
          <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.15" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const y = PAD.top + t * (H - PAD.top - PAD.bottom);
          const val = maxV - t * range;
          return (
            <g key={t}>
              <line x1={PAD.left} y1={y} x2={W - PAD.right} y2={y} stroke="#27272a" strokeWidth="1" />
              <text x={PAD.left - 6} y={y + 4} textAnchor="end" fill="#52525b" fontSize="9" fontFamily="monospace">
                ${(val / 1000).toFixed(0)}k
              </text>
            </g>
          );
        })}

        {/* X labels */}
        {labels.map((idx) => (
          <text key={idx} x={pts[idx].x} y={H - 4} textAnchor="middle" fill="#52525b" fontSize="9" fontFamily="monospace">
            {pts[idx].date}
          </text>
        ))}

        {/* Area fill */}
        <path d={areaPath} fill="url(#areaGrad)" />

        {/* Line */}
        <path d={linePath} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />

        {/* Hover overlay */}
        {pts.map((p, i) => (
          <rect
            key={i}
            x={p.x - (W / DATA.length) / 2}
            y={PAD.top}
            width={W / DATA.length}
            height={H - PAD.top - PAD.bottom}
            fill="transparent"
            onMouseEnter={() => setHover(i)}
          />
        ))}

        {/* Hover indicator */}
        {hoverPt && (
          <>
            <line x1={hoverPt.x} y1={PAD.top} x2={hoverPt.x} y2={H - PAD.bottom} stroke="#52525b" strokeWidth="1" strokeDasharray="3,3" />
            <circle cx={hoverPt.x} cy={hoverPt.y} r="3" fill={color} />
            <rect x={Math.min(hoverPt.x - 40, W - 110)} y={PAD.top} width={100} height={30} rx="3" fill="#18181b" stroke="#3f3f46" strokeWidth="1" />
            <text x={Math.min(hoverPt.x - 40, W - 110) + 50} y={PAD.top + 11} textAnchor="middle" fill="#a1a1aa" fontSize="8" fontFamily="monospace">
              {hoverPt.date}
            </text>
            <text x={Math.min(hoverPt.x - 40, W - 110) + 50} y={PAD.top + 23} textAnchor="middle" fill={color} fontSize="9" fontFamily="monospace" fontWeight="bold">
              ${hoverPt.value.toLocaleString()}
            </text>
          </>
        )}
      </svg>

      {/* Stats row */}
      <div className="grid grid-cols-4 gap-3 pt-2 border-t border-zinc-800">
        {[
          { label: 'Start', value: `$${(pts[0].value / 1000).toFixed(1)}k` },
          { label: 'Current', value: `$${(pts[pts.length - 1].value / 1000).toFixed(1)}k` },
          { label: 'Peak', value: `$${(Math.max(...values) / 1000).toFixed(1)}k` },
          { label: 'Return', value: `${isProfit ? '+' : ''}${totalReturn}%`, color: isProfit ? 'text-emerald-400' : 'text-red-400' },
        ].map((s) => (
          <div key={s.label} className="flex flex-col">
            <span className="text-[9px] uppercase tracking-widest text-zinc-600 font-mono">{s.label}</span>
            <span className={`text-sm font-light tabular-nums font-mono ${s.color || 'text-zinc-300'}`}>{s.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
