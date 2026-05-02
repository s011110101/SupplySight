import { useState } from 'react';
import {
  ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer,
} from 'recharts';
import type { TrendPointDTO } from '../types/dashboard';

interface TrendVisualizationProps {
  points: TrendPointDTO[] | null;
  loading?: boolean;
  showHealthIndex?: boolean;
}

type View = 'risk' | 'imports' | 'price';

const ALL_VIEWS: { key: View; label: string }[] = [
  { key: 'risk', label: 'Supply Health Index' },
  { key: 'imports', label: 'Monthly Imports' },
  { key: 'price', label: 'Price Index' },
];

const ALERTS: { date: string; url: string; title: string }[] = [
  {
    date: '2025-08',
    url: 'https://www.fda.gov/safety/major-product-recalls/2025-recalls-frozen-shrimp-products-associated-cesium-137-contamination-pt-bahari-makmur-sejati-due',
    title: '2025 Recalls of Frozen Shrimp Products Associated with Cesium-137 Contamination from PT. Bahari Makmur Sejati Due to Potential Safety Concerns',
  },
  {
    date: '2026-03',
    url: 'https://seafoodnews.com/Story/1336206/Crude-Oil-Set-for-Record-Daily-Surge-as-Iran-War-Enters-Tenth-Day',
    title: 'Crude Oil Set for Record Daily Surge as Iran War Enters Tenth Day',
  },
];

function alertForDate(date: string) {
  return ALERTS.find((a) => a.date === date) ?? null;
}

function fmtImport(v: number) {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M lbs`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(0)}K lbs`;
  return `${v} lbs`;
}

function shiLabel(shi: number) {
  if (shi >= 7.5) return 'Healthy';
  if (shi >= 5.0) return 'Moderate';
  if (shi >= 2.5) return 'At Risk';
  return 'Critical';
}

function AlertDot(props: any) {
  const { cx, cy, payload } = props;
  if (!cx || !cy) return null;
  if (alertForDate(payload?.date)) {
    return (
      <g>
        <circle
          cx={cx} cy={cy} r={7}
          fill="#ef4444" stroke="white" strokeWidth={2}
          style={{ cursor: 'pointer' }}
        />
        <text
          x={cx} y={cy + 1}
          textAnchor="middle" dominantBaseline="middle"
          fill="white" fontSize={9} fontWeight="bold"
          style={{ pointerEvents: 'none' }}
        >!</text>
      </g>
    );
  }
  return <circle cx={cx} cy={cy} r={2} fill="#3b82f6" />;
}

interface PinnedPoint {
  date: string;
  value: number;
  x: number;
  y: number;
}

function PinnedTooltip({ point, onClose }: { point: PinnedPoint; onClose: () => void }) {
  const alert = alertForDate(point.date);
  return (
    <div
      style={{
        position: 'absolute',
        left: point.x,
        top: point.y,
        transform: 'translate(-50%, calc(-100% - 14px))',
        zIndex: 20,
        pointerEvents: 'auto',
        maxWidth: '420px',
      }}
    >
      <div className="bg-white border border-slate-300 rounded-lg shadow-lg p-3" style={{ fontSize: '12px' }}>
        <div className="flex items-center justify-between gap-4 mb-1">
          <span className="text-slate-500">{point.date}</span>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600 leading-none">✕</button>
        </div>
        <p className="font-medium text-slate-800">
          {point.value.toFixed(1)} ({shiLabel(point.value)})
        </p>
        {alert && (
          <a
            href={alert.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 underline block mt-2 leading-snug"
            style={{ fontSize: '11px' }}
          >
            ⚠ {alert.title}
          </a>
        )}
      </div>
      <div
        style={{
          position: 'absolute',
          bottom: -6,
          left: '50%',
          transform: 'translateX(-50%)',
          width: 0,
          height: 0,
          borderLeft: '6px solid transparent',
          borderRight: '6px solid transparent',
          borderTop: '6px solid #cbd5e1',
        }}
      />
    </div>
  );
}

function HoverTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const value: number = payload[0]?.value;
  return (
    <div style={{ backgroundColor: 'white', border: '1px solid #e2e8f0', borderRadius: '6px', fontSize: '12px', padding: '8px 12px' }}>
      <p className="text-slate-500 mb-1">{label}</p>
      <p className="font-medium text-slate-800">{value?.toFixed(1)} ({shiLabel(value)})</p>
      {alertForDate(label) && (
        <p className="text-red-500 text-xs mt-1">⚠ Click to see alert info</p>
      )}
    </div>
  );
}

export function TrendVisualization({ points, loading, showHealthIndex = true }: TrendVisualizationProps) {
  const views = showHealthIndex ? ALL_VIEWS : ALL_VIEWS.filter((v) => v.key !== 'risk');
  const [view, setView] = useState<View>(showHealthIndex ? 'risk' : 'imports');
  const [pinnedPoint, setPinnedPoint] = useState<PinnedPoint | null>(null);

  if (loading) {
    return (
      <div className="bg-white border border-slate-200 rounded-lg animate-pulse">
        <div className="p-5 border-b border-slate-200 space-y-2">
          <div className="h-5 w-56 bg-slate-100 rounded" />
          <div className="h-4 w-80 bg-slate-100 rounded" />
        </div>
        <div className="p-5 h-[320px] bg-slate-50" />
      </div>
    );
  }

  const FORECAST: { date: string; risk: number }[] = [
    { date: '2026-03', risk: 3.1 },
    { date: '2026-04', risk: 3.8 },
    { date: '2026-05', risk: 2.7 },
    { date: '2026-06', risk: 3.1},
  ];

  const apiData = (points ?? []).map((p) => ({
    date: p.date,
    risk: p.shrimp != null ? (100 - p.shrimp) / 10 : undefined,
    riskDashed: undefined as number | undefined,
    imports: p.monthlyImport ?? undefined,
    price: p.priceIndex ?? undefined,
  }));

  const forecastData = FORECAST.map(({ date, risk }, i) => ({
    date,
    risk: i < FORECAST.length - 2 ? risk : undefined,
    riskDashed: i >= FORECAST.length - 3 ? risk : undefined,
    imports: undefined as number | undefined,
    price: undefined as number | undefined,
  }));

  const data = [...apiData, ...forecastData];

  const xTicks = data.map((d) => d.date).filter((_, i) => i % 6 === 0);

  function handleChartClick(payload: any) {
    if (view !== 'risk' || !payload?.activePayload?.length) return;
    const date = payload.activeLabel as string;
    const value = payload.activePayload[0]?.value as number;
    if (pinnedPoint?.date === date) {
      setPinnedPoint(null);
    } else {
      setPinnedPoint({ date, value, x: payload.chartX, y: payload.chartY });
    }
  }

  return (
    <div className="bg-white border border-slate-200 rounded-lg">
      <div className="p-5 border-b border-slate-200 flex items-start justify-between">
        <div>
          <h2 className="text-slate-900 mb-1">Trends Over Time</h2>
          <p className="text-slate-600 text-sm">Monthly data points</p>
        </div>
        <div className="flex gap-1">
          {views.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => { setView(key); setPinnedPoint(null); }}
              className={`px-3 py-1.5 text-xs rounded-lg border transition-colors ${
                view === key
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'text-slate-600 border-slate-200 hover:border-blue-300 hover:bg-slate-50'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="p-5 relative">
        {data.length === 0 ? (
          <div className="h-[320px] flex items-center justify-center text-slate-500 text-sm border border-dashed border-slate-200 rounded-lg">
            No trend data available.
          </div>
        ) : (
          <>
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart data={data} margin={{ left: 10, right: 10 }} onClick={handleChartClick}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="date" ticks={xTicks} stroke="#64748b" style={{ fontSize: '11px' }} />

                {view === 'risk' && (
                  <>
                    <YAxis domain={[0, 10]} stroke="#64748b" style={{ fontSize: '11px' }} />
                    <Tooltip content={<HoverTooltip />} />
                    <Line
                      type="monotone"
                      dataKey="risk"
                      name="Supply Health Index"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={<AlertDot />}
                      activeDot={{ r: 4, style: { cursor: 'pointer' } }}
                      connectNulls={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="riskDashed"
                      name="Forecast"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      strokeDasharray="5 4"
                      dot={{ r: 2, fill: '#3b82f6' }}
                      activeDot={{ r: 4 }}
                      connectNulls={false}
                      legendType="none"
                    />
                  </>
                )}

                {view === 'imports' && (
                  <>
                    <YAxis stroke="#64748b" style={{ fontSize: '11px' }} tickFormatter={(v) => `${(v / 1_000_000).toFixed(0)}M`} />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'white', border: '1px solid #e2e8f0', borderRadius: '6px', fontSize: '12px' }}
                      formatter={(value: number) => [fmtImport(value), 'Monthly Import']}
                    />
                    <Bar dataKey="imports" name="Monthly Import" fill="#3b82f6" opacity={0.8} radius={[2, 2, 0, 0]} />
                  </>
                )}

                {view === 'price' && (
                  <>
                    <YAxis stroke="#64748b" style={{ fontSize: '11px' }} domain={['auto', 'auto']} />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'white', border: '1px solid #e2e8f0', borderRadius: '6px', fontSize: '12px' }}
                      formatter={(value: number) => [value.toFixed(1), 'Price Index']}
                    />
                    <Line type="monotone" dataKey="price" name="Price Index" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 2 }} activeDot={{ r: 4 }} connectNulls={false} />
                  </>
                )}

                <Legend wrapperStyle={{ fontSize: '12px' }} />
              </ComposedChart>
            </ResponsiveContainer>

            {pinnedPoint && view === 'risk' && (
              <PinnedTooltip point={pinnedPoint} onClose={() => setPinnedPoint(null)} />
            )}
          </>
        )}
      </div>
    </div>
  );
}
