import { Shield, Brain, Database, TrendingDown, DollarSign, BarChart3, Droplets } from 'lucide-react';

const SHI_BANDS = [
  {
    label: 'Critical',
    range: '0.0 to 2.5',
    color: 'text-red-700',
    bg: 'bg-red-50',
    border: 'border-red-200',
    description: 'High disruption risk. Place an emergency order and find backup suppliers.',
  },
  {
    label: 'At Risk',
    range: '2.6 to 5.0',
    color: 'text-orange-700',
    bg: 'bg-orange-50',
    border: 'border-orange-200',
    description: 'Supply is tight. Order more before the next cycle.',
  },
  {
    label: 'Moderate',
    range: '5.1 to 7.5',
    color: 'text-yellow-700',
    bg: 'bg-yellow-50',
    border: 'border-yellow-200',
    description: 'Some stress in the data. Check weekly and order a bit early if it gets worse.',
  },
  {
    label: 'Healthy',
    range: '7.6 to 10.0',
    color: 'text-green-700',
    bg: 'bg-green-50',
    border: 'border-green-200',
    description: 'Supply is stable. No action needed.',
  },
];

const FEATURES = [
  { icon: TrendingDown, label: 'Monthly import volume', detail: 'Total US shrimp imports (HS 030616 + 030617) from the Census Bureau. Lower volumes lower the index.' },
  { icon: BarChart3, label: '6-month import z-score', detail: 'How many standard deviations current volume is above or below the 6-month mean. Negative z-scores lower the index the most.' },
  { icon: BarChart3, label: '3-month import std dev', detail: 'Rolling volatility of import volumes. Higher volatility lowers the index.' },
  { icon: TrendingDown, label: 'Month-over-month change', detail: 'Percentage change in imports vs. last month. Sharp drops lower the index further.' },
  { icon: TrendingDown, label: 'Year-over-year change', detail: 'Long-run import trend, used for seasonal correction.' },
  { icon: DollarSign, label: 'FAO price index', detail: 'FAO shrimp price index value. High prices (above the training median around 80) lower the index.' },
  { icon: Droplets, label: 'Oil price (daily adjustment)', detail: 'Brent crude oil price used in the formula adjustment. Higher oil raises shipping costs and can shift the index by up to 1.5 points.' },
];

const RULES = [
  {
    id: 1,
    icon: Shield,
    name: 'At Risk Threshold',
    description: 'Index ≤ 5.0 triggers an At Risk alert. Order more before the next cycle.',
    category: 'Model Rule',
    status: 'Active',
  },
  {
    id: 2,
    icon: Shield,
    name: 'Critical Threshold',
    description: 'Index ≤ 2.5 triggers a Critical alert. Place an emergency order and contact backup suppliers within 2 days.',
    category: 'Model Rule',
    status: 'Active',
  },
  {
    id: 3,
    icon: Shield,
    name: 'Negative z-score signal',
    description: 'Import volume below the 6-month average (z-score < 0) is the strongest predictor of supply stress and the main driver pushing the index down.',
    category: 'Feature Rule',
    status: 'Active',
  },
  {
    id: 4,
    icon: Shield,
    name: 'Price stress signal',
    description: 'When the FAO shrimp price index goes above the training median (around 80), the excess pushes the index down.',
    category: 'Feature Rule',
    status: 'Active',
  },
];

export function Rules() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-slate-900 mb-1">Metrics</h1>
        <p className="text-slate-600">How SupplySight calculates the Supply Health Index for shrimp</p>
      </div>

      <div className="bg-white border border-blue-200 rounded-xl p-6 space-y-5">
        <div className="flex items-center gap-3">
          <div className="bg-blue-50 p-2.5 rounded-lg">
            <Brain className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <h2 className="text-slate-900 text-lg font-semibold">Supply Health Index Model</h2>
            <p className="text-slate-500 text-xs">Version 8 · trained through Jan 2025 · architecture: monthly linear head + oil/sentiment formula adjustment</p>
          </div>
        </div>

        <p className="text-slate-600 text-sm leading-relaxed">
          Every month, the model uses the latest US shrimp import data and the FAO price index to compute a
          {' '}<strong>Supply Health Index from 0 to 10</strong>. <strong>10 is fully healthy</strong> and
          {' '}<strong>0 indicates high supply disruption risk</strong>. The index comes from a linear regression
          head with a small deterministic <strong>formula adjustment</strong> driven by oil price. The final
          value sets the health band shown on the dashboard.
        </p>

        <div>
          <h3 className="text-slate-800 font-medium text-sm mb-3">Supply Health Index bands</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {SHI_BANDS.map((b) => (
              <div key={b.label} className={`${b.bg} border ${b.border} rounded-lg p-3`}>
                <div className="flex items-center justify-between mb-1">
                  <span className={`font-semibold text-sm ${b.color}`}>{b.label}</span>
                  <span className={`text-xs font-mono ${b.color}`}>{b.range}</span>
                </div>
                <p className="text-xs text-slate-600 leading-snug">{b.description}</p>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-slate-800 font-medium text-sm mb-3">Input features</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {FEATURES.map((f) => (
              <div key={f.label} className="flex items-start gap-3 bg-slate-50 border border-slate-100 rounded-lg p-3">
                <f.icon className="w-4 h-4 text-slate-400 mt-0.5 shrink-0" />
                <div>
                  <p className="text-slate-800 text-xs font-medium">{f.label}</p>
                  <p className="text-slate-500 text-xs leading-snug mt-0.5">{f.detail}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="flex items-start gap-3 bg-slate-50 border border-slate-100 rounded-lg p-4">
          <Database className="w-4 h-4 text-slate-400 mt-0.5 shrink-0" />
          <div>
            <p className="text-slate-800 text-xs font-semibold mb-0.5">Data sources</p>
            <p className="text-slate-500 text-xs leading-relaxed">
              <strong>US Census Bureau</strong>: monthly international trade imports (HS codes 030616 &amp; 030617, frozen shrimp/prawns).<br />
              <strong>FAO</strong>: Fish Price Index (shrimp component), monthly.<br />
              <strong>Brent crude oil</strong>: daily spot price used for the formula adjustment.
            </p>
          </div>
        </div>
      </div>

      <div>
        <h2 className="text-slate-900 font-semibold mb-4">Alert Rules</h2>
        <div className="grid grid-cols-1 gap-4">
          {RULES.map((rule) => (
            <div key={rule.id} className="bg-white border border-slate-200 rounded-lg p-6">
              <div className="flex items-start gap-4">
                <div className="bg-blue-50 p-3 rounded-lg">
                  <rule.icon className="w-6 h-6 text-blue-600" />
                </div>
                <div className="flex-1">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <h3 className="text-slate-900 font-medium mb-1">{rule.name}</h3>
                      <p className="text-slate-600 text-sm">{rule.description}</p>
                    </div>
                    <span className="px-3 py-1 bg-green-50 text-green-700 text-xs font-medium rounded-full border border-green-200">
                      {rule.status}
                    </span>
                  </div>
                  <div className="text-xs text-slate-500 mt-3">Category: {rule.category}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
