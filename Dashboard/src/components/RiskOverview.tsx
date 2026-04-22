import { useState } from 'react';
import { AlertTriangle, Package, Map, TrendingUp, TrendingDown, CheckCircle } from 'lucide-react';
import type { OverviewMetricDTO } from '../types/dashboard';
import { RiskBreakdownModal } from './RiskBreakdownModal';

function shiStatusLabel(value: string): string {
  const shi = parseFloat(value);
  if (isNaN(shi)) return '';
  if (shi >= 7.5) return 'Healthy';
  if (shi >= 5.0) return 'Moderate';
  if (shi >= 2.5) return 'At Risk';
  return 'Critical';
}

function riskStyles(value: string) {
  const shi = parseFloat(value);
  if (!isNaN(shi)) {
    if (shi <= 2.5) return { color: 'text-red-600', bgColor: 'bg-red-50', borderColor: 'border-red-200' };
    if (shi <= 5.0) return { color: 'text-orange-600', bgColor: 'bg-orange-50', borderColor: 'border-orange-200' };
    if (shi <= 7.5) return { color: 'text-yellow-600', bgColor: 'bg-yellow-50', borderColor: 'border-yellow-200' };
    return { color: 'text-green-600', bgColor: 'bg-green-50', borderColor: 'border-green-200' };
  }
  return { color: 'text-slate-600', bgColor: 'bg-slate-50', borderColor: 'border-slate-200' };
}

function metricVisual(metric: OverviewMetricDTO) {
  if (metric.key === 'risk') {
    const s = riskStyles(metric.value);
    return { Icon: AlertTriangle, ...s };
  }
  if (metric.key === 'products') {
    return {
      Icon: Map,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200',
    };
  }
  return {
    Icon: Package,
    color: 'text-blue-600',
    bgColor: 'bg-blue-50',
    borderColor: 'border-blue-200',
  };
}

const LABEL_MAP: Record<string, string> = {
  price_index_value: 'Price Index',
  oil_price: 'Oil Price',
  sentiment_score: 'Sentiment Score',
  import_volume: 'Import Volume',
  relevancy_score: 'Relevancy Score',
};

function humanize(key: string): string {
  return LABEL_MAP[key] ?? key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

function anomalyCardStyles(entries: [string, number][]) {
  if (entries.length === 0)
    return { color: 'text-green-600', bgColor: 'bg-green-50', borderColor: 'border-green-200' };
  const maxAbs = Math.max(...entries.map(([, z]) => Math.abs(z)));
  if (maxAbs >= 2.5) return { color: 'text-red-600', bgColor: 'bg-red-50', borderColor: 'border-red-200' };
  if (maxAbs >= 2.0) return { color: 'text-orange-600', bgColor: 'bg-orange-50', borderColor: 'border-orange-200' };
  return { color: 'text-yellow-600', bgColor: 'bg-yellow-50', borderColor: 'border-yellow-200' };
}

interface RiskOverviewProps {
  metrics: OverviewMetricDTO[] | null;
  anomalies?: Record<string, number>;
  loading?: boolean;
}

export function RiskOverview({ metrics, anomalies, loading }: RiskOverviewProps) {
  const [breakdownOpen, setBreakdownOpen] = useState(false);

  if (loading) {
    return (
      <div className="grid grid-cols-3 gap-4">
        {[0, 1, 2].map((i) => (
          <div key={i} className="bg-white border border-slate-200 rounded-lg p-5 animate-pulse">
            <div className="h-9 w-9 bg-slate-100 rounded-lg mb-3" />
            <div className="h-4 w-32 bg-slate-100 rounded mb-2" />
            <div className="h-8 w-20 bg-slate-100 rounded mb-2" />
            <div className="h-3 w-full bg-slate-100 rounded" />
          </div>
        ))}
      </div>
    );
  }

  if (!metrics?.length) {
    return (
      <div className="bg-white border border-dashed border-slate-200 rounded-lg p-6 text-slate-600 text-sm">
        No overview metrics (database may be empty or unreachable).
      </div>
    );
  }

  const anomalyEntries = Object.entries(anomalies ?? {}).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const { color: aColor, bgColor: aBg, borderColor: aBorder } = anomalyCardStyles(anomalyEntries);
  const [topKey, topZ] = anomalyEntries[0] ?? [null, null];
  const riskBreakdown = metrics.find((m) => m.key === 'risk')?.breakdown ?? null;

  return (
    <div className="grid grid-cols-3 gap-4">
      {metrics.map((metric) => {
        const { Icon, color, bgColor, borderColor } = metricVisual(metric);
        const inner = (
          <>
            <div className="flex items-start justify-between mb-3">
              <div className={`${bgColor} p-2 rounded-lg`}>
                <Icon className={`w-5 h-5 ${color}`} />
              </div>
              {metric.key === 'risk' && (
                <span className="text-xs text-slate-400 font-medium">View breakdown ›</span>
              )}
            </div>
            <div className="space-y-1">
              <p className="text-slate-600 text-sm">{metric.label}</p>
              <p className={`text-3xl font-semibold ${color}`}>
                {metric.value}
                {metric.key === 'risk' && shiStatusLabel(metric.value) && (
                  <span className="text-base font-medium ml-2">{shiStatusLabel(metric.value)}</span>
                )}
              </p>
              <p className="text-slate-500 text-xs">{metric.subtext}</p>
            </div>
          </>
        );

        if (metric.key === 'risk') {
          return (
            <button
              key={metric.key}
              type="button"
              onClick={() => setBreakdownOpen(true)}
              className={`bg-white border ${borderColor} rounded-lg p-5 block hover:shadow-md transition-shadow cursor-pointer text-left w-full`}
            >
              {inner}
            </button>
          );
        }

        return (
          <div key={metric.key} className={`bg-white border ${borderColor} rounded-lg p-5`}>
            {inner}
          </div>
        );
      })}

      {/* Anomalies card */}
      <div className={`bg-white border ${aBorder} rounded-lg p-5`}>
        <div className="flex items-start justify-between mb-3">
          <div className={`${aBg} p-2 rounded-lg`}>
            <AlertTriangle className={`w-5 h-5 ${aColor}`} />
          </div>
        </div>
        <div className="space-y-1">
          <p className="text-slate-600 text-sm">Active Anomalies</p>
          <p className={`text-3xl font-semibold ${aColor}`}>{anomalyEntries.length}</p>
          {anomalyEntries.length === 0 ? (
            <div className="flex items-center gap-1 text-xs text-green-700">
              <CheckCircle className="w-3 h-3" />
              <span>All metrics within normal range</span>
            </div>
          ) : (
            <div className="space-y-1 mt-1">
              {anomalyEntries.slice(0, 3).map(([key, z]) => (
                <div key={key} className="flex items-center gap-1 text-xs text-slate-600">
                  {z > 0
                    ? <TrendingUp className="w-3 h-3 text-red-500 shrink-0" />
                    : <TrendingDown className="w-3 h-3 text-blue-500 shrink-0" />}
                  <span className="truncate">{humanize(key)}</span>
                  <span className={`ml-auto font-medium tabular-nums ${aColor}`}>
                    {z > 0 ? '+' : ''}{z.toFixed(1)}σ
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {breakdownOpen && (
        <RiskBreakdownModal
          breakdown={riskBreakdown}
          onClose={() => setBreakdownOpen(false)}
        />
      )}
    </div>
  );
}
