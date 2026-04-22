import { useEffect } from 'react';
import { X, TrendingUp, TrendingDown } from 'lucide-react';
import type { RiskBreakdownDTO } from '../types/dashboard';

interface RiskBreakdownModalProps {
  breakdown: RiskBreakdownDTO | null | undefined;
  onClose: () => void;
}

function fmtValue(v: number | null): string {
  if (v === null) return '—';
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return v.toLocaleString(undefined, { maximumFractionDigits: 3 });
}

export function RiskBreakdownModal({ breakdown, onClose }: RiskBreakdownModalProps) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  if (!breakdown) {
    return (
      <div
        className="fixed inset-0 bg-slate-900/40 z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <div className="bg-white rounded-lg p-6 max-w-md text-slate-700" onClick={(e) => e.stopPropagation()}>
          No breakdown available — the risk model isn't loaded on the backend.
        </div>
      </div>
    );
  }

  // Convert from model's risk-score space (0–100, higher = worse) to the SHI
  // space shown on the dashboard card (0–10, higher = better):
  //   shi = (100 - score) / 10
  // For contributions the sign flips and magnitude divides by 10:
  //   shi_contribution = -contribution / 10
  const toShi = (score: number) => (100 - score) / 10;
  const baselineShi = toShi(breakdown.baseline);
  const monthlyShi = toShi(breakdown.monthlyPrediction);
  const dailyShi = -breakdown.dailyAdjustment / 10;
  const finalShi = toShi(breakdown.finalScore);

  const shiContribs = breakdown.contributions.map((c) => ({
    ...c,
    shiDelta: -c.contribution / 10,
  }));
  const maxAbs = Math.max(1e-6, ...shiContribs.map((c) => Math.abs(c.shiDelta)));

  return (
    <div
      className="fixed inset-0 bg-slate-900/40 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between p-5 border-b border-slate-200">
          <div>
            <h2 className="text-lg font-medium text-slate-900">Supply Health Index breakdown</h2>
            <p className="text-sm text-slate-500 mt-0.5">
              How each input moves health away from the model baseline (0–10 scale, higher = healthier).
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-slate-100 rounded-md text-slate-500"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Summary */}
        <div className="grid grid-cols-4 gap-3 p-5 border-b border-slate-200">
          <div>
            <div className="text-xs text-slate-500">Baseline SHI</div>
            <div className="text-lg font-semibold text-slate-700 tabular-nums">
              {baselineShi.toFixed(1)}
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-500">After monthly inputs</div>
            <div className="text-lg font-semibold text-slate-700 tabular-nums">
              {monthlyShi.toFixed(1)}
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-500">Daily adj.</div>
            <div className="text-lg font-semibold text-slate-700 tabular-nums">
              {dailyShi >= 0 ? '+' : ''}{dailyShi.toFixed(1)}
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-500">Final SHI</div>
            <div className="text-lg font-semibold text-slate-900 tabular-nums">
              {finalShi.toFixed(1)}
              <span className="text-xs font-medium text-slate-500 ml-2">{breakdown.level}</span>
            </div>
          </div>
        </div>

        {/* Contributions */}
        <div className="p-5 space-y-3">
          <div className="text-xs text-slate-500 uppercase tracking-wide">
            Feature contributions to SHI
          </div>
          {shiContribs.map((c) => {
            const pullsHealthDown = c.shiDelta < 0;
            const pct = (Math.abs(c.shiDelta) / maxAbs) * 100;
            return (
              <div key={c.feature} className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-1.5 text-slate-700">
                    {pullsHealthDown
                      ? <TrendingDown className="w-3.5 h-3.5 text-red-500" />
                      : <TrendingUp className="w-3.5 h-3.5 text-green-600" />}
                    <span>{c.label}</span>
                    {c.imputed && (
                      <span className="text-xs text-slate-400 ml-1">(imputed)</span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 tabular-nums">
                    <span className="text-xs text-slate-500">{fmtValue(c.value)}</span>
                    <span className={`text-sm font-medium ${pullsHealthDown ? 'text-red-600' : 'text-green-700'}`}>
                      {c.shiDelta >= 0 ? '+' : ''}{c.shiDelta.toFixed(2)}
                    </span>
                  </div>
                </div>
                <div className="relative h-1.5 bg-slate-100 rounded overflow-hidden">
                  <div
                    className={`absolute top-0 bottom-0 ${pullsHealthDown ? 'bg-red-400' : 'bg-green-500'}`}
                    style={{
                      left: pullsHealthDown ? `${50 - pct / 2}%` : '50%',
                      width: `${pct / 2}%`,
                    }}
                  />
                  <div className="absolute top-0 bottom-0 w-px bg-slate-300" style={{ left: '50%' }} />
                </div>
              </div>
            );
          })}
        </div>

        <div className="px-5 pb-5 text-xs text-slate-500">
          Red bars pull the health score down; green bars push it up.
          Contributions sum exactly to <span className="tabular-nums">(After monthly inputs − Baseline SHI)</span>.
        </div>
      </div>
    </div>
  );
}
