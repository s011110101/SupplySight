import { useState } from 'react';
import { ChevronDown, Package } from 'lucide-react';
import { RiskOverview } from './RiskOverview';
import { TrendVisualization } from './TrendVisualization';
import { EvidencePanel } from './EvidencePanel';
import { DecisionSupportPanel } from './DecisionSupportPanel';
import { useDashboard } from '../hooks/useDashboard';

export function Dashboard() {
  const { data, loading, error } = useDashboard();
  const [selectorOpen, setSelectorOpen] = useState(false);

  const products = data?.products ?? [];
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selectedProduct = products.find((p) => p.id === selectedId) ?? products[0] ?? null;

  return (
    <div className="space-y-6">
      <div className="flex items-start gap-4">
        {/* Product selector */}
        <div className="relative mt-1">
          <button
            onClick={() => setSelectorOpen((o) => !o)}
            className="flex items-center gap-2 bg-white border border-slate-200 rounded-lg px-4 py-2.5 text-sm text-slate-700 hover:border-blue-300 hover:bg-slate-50 transition-colors shadow-sm"
          >
            <Package className="w-4 h-4 text-blue-600" />
            <span className="font-medium">
              {loading ? 'Loading…' : selectedProduct?.name ?? 'Select product'}
            </span>
            <ChevronDown className="w-4 h-4 text-slate-400" />
          </button>

          {selectorOpen && products.length > 0 && (
            <div className="absolute right-0 mt-1 w-64 bg-white border border-slate-200 rounded-lg shadow-lg z-20">
              {products.map((p) => (
                <button
                  key={p.id}
                  onClick={() => { setSelectedId(p.id); setSelectorOpen(false); }}
                  className={`w-full text-left px-4 py-3 text-sm hover:bg-slate-50 first:rounded-t-lg last:rounded-b-lg transition-colors ${
                    p.id === (selectedProduct?.id) ? 'text-blue-700 bg-blue-50' : 'text-slate-700'
                  }`}
                >
                  <div className="font-medium">{p.name}</div>
                  <div className="text-xs text-slate-500">{p.category}</div>
                </button>
              ))}
            </div>
          )}
        </div>

        <div>
          <h1 className="text-slate-900 mb-1">Risk Dashboard</h1>
          <p className="text-slate-600">Monitor and forecast supply chain risks across your product portfolio</p>
          {error && (
            <div className="mt-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800" role="alert">
              {error}
            </div>
          )}
          {data?.meta?.placeholderReason === 'database_unavailable' && data.meta.dbError && (
            <div className="mt-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-900" role="alert">
              <span className="font-medium">Unable to connect to database.</span>
              <span className="block text-xs mt-1 font-mono break-all">{data.meta.dbError}</span>
            </div>
          )}
          {data?.meta && (
            <p className="text-slate-500 text-xs mt-2">
              Data as of: {data.meta.asOf ?? '—'} · Generated: {data.meta.generatedAt}
            </p>
          )}
        </div>
      </div>

      <RiskOverview metrics={data?.overview ?? null} loading={loading} />

      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-2 space-y-6">
          <TrendVisualization points={data?.trend ?? null} loading={loading} />
          <DecisionSupportPanel recommendations={data?.recommendations ?? null} loading={loading} />
        </div>
        <div className="col-span-1 space-y-6">
          <EvidencePanel items={data?.evidence ?? null} loading={loading} />
        </div>
      </div>
    </div>
  );
}
