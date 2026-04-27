import { useState, useEffect } from 'react';
import { ChevronDown, Package } from 'lucide-react';
import { RiskOverview } from './RiskOverview';
import { TrendVisualization } from './TrendVisualization';
import { EvidencePanel } from './EvidencePanel';
import { DecisionSupportPanel } from './DecisionSupportPanel';
import { useDashboard } from '../hooks/useDashboard';

export function Dashboard() {
  const { data, loading, error } = useDashboard();

  const [selectorOpen, setSelectorOpen] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const products = [
      { id: "shrimp", name: "Shrimp", category: "Seafood" },
    { id: "salmon", name: "Salmon", category: "Seafood" },
    { id: "tuna", name: "Tuna", category: "Seafood" },
    { id: "whitefish", name: "Whitefish", category: "Seafood" },
  ];
  const selectedProduct = products.find((p) => p.id === selectedId) ?? products[0] ?? null;

  const [simpleData, setSimpleData] = useState<any[]>([]);
  const [simpleLoading, setSimpleLoading] = useState(false);

  const productName = selectedProduct?.name?.toLowerCase() || "shrimp";

  const simpleTrend = simpleData.map((d) => ({
    date: d.date as string,
    monthlyImport: d.monthly_import as number | null,
    priceIndex: d.price_index_value as number | null,
  }));

  useEffect(() => {
    if (!productName || productName.includes("shrimp")) return;

    const p = productName.split(" ")[0];

    setSimpleLoading(true);
    fetch(`/api/monthly?product=${p}`)
    .then(res => res.json())
      .then(res => {
        setSimpleData(res);
        setSimpleLoading(false);
      })
      .catch(() => setSimpleLoading(false));
  }, [productName]);

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-slate-900 mb-1">Dashboard</h1>
          <p className="text-slate-600">Monitor and forecast supply chain risks</p>

          {error && (
            <div className="mt-3 border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
              {error}
            </div>
          )}
        </div>

        {/* Product selector */}
        <div className="relative mt-1">
          <button
            onClick={() => setSelectorOpen((o) => !o)}
            className="flex items-center gap-2 bg-white border border-slate-200 rounded-lg px-4 py-2.5 text-sm"
          >
            <Package className="w-4 h-4 text-blue-600" />
            <span>
              {loading ? 'Loading…' : selectedProduct?.name ?? 'Select product'}
            </span>
            <ChevronDown className="w-4 h-4 text-slate-400" />
          </button>

          {selectorOpen && products.length > 0 && (
            <div className="absolute right-0 mt-1 w-64 bg-white border rounded-lg shadow-lg z-20">
              {products.map((p) => (
                <button
                  key={p.id}
                  onClick={() => { setSelectedId(p.id); setSelectorOpen(false); }}
                  className="w-full text-left px-4 py-3 text-sm hover:bg-slate-50"
                >
                  <div>{p.name}</div>
                  <div className="text-xs text-slate-500">{p.category}</div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {productName.includes("shrimp") ? (
        <>
          <RiskOverview metrics={data?.overview ?? null} anomalies={data?.anomalies} loading={loading} />

          <div className="grid grid-cols-3 gap-6">
            <div className="col-span-2 space-y-6">
              <TrendVisualization points={data?.trend ?? null} loading={loading} />
              <DecisionSupportPanel recommendations={data?.recommendations ?? null} loading={loading} />
            </div>
            <div className="col-span-1 space-y-6">
              <EvidencePanel items={data?.evidence ?? null} loading={loading} />
            </div>
          </div>
        </>
      ) : (
        <div className="space-y-6">
          <TrendVisualization
            points={simpleTrend}
            loading={simpleLoading}
            showHealthIndex={false}
          />
        </div>
      )}
    </div>
  );
}