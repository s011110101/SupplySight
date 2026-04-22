/**
 * Mirrors `GET /api/dashboard` from `services/api/main.py`.
 *
 * UNCERTAIN (confirm with backend / domain):
 * - `trend[].shrimp` is a heuristic 0–100 score from `monthly_import_zscore_6`, not a stored model.
 * - `products[].risk30/risk60/risk90` are rolling monthly windows, not true 30/60/90-day forecasts.
 * - `evidence` is synthesized from `dates_shrimp` fields until a news/articles API exists.
 */
export type RiskTrend = 'up' | 'down' | 'stable';

export interface ProductRiskHorizon {
  level: string;
  score: number;
  trend: RiskTrend;
}

export interface DashboardProductRow {
  id: string;
  name: string;
  category: string;
  supplier: string;
  risk30: ProductRiskHorizon;
  risk60: ProductRiskHorizon;
  risk90: ProductRiskHorizon;
}

export interface RiskContributionDTO {
  feature: string;
  label: string;
  value: number | null;
  imputed: boolean;
  contribution: number;
}

export interface RiskBreakdownDTO {
  baseline: number;
  monthlyPrediction: number;
  dailyAdjustment: number;
  finalScore: number;
  level: string;
  contributions: RiskContributionDTO[];
}

export interface OverviewMetricDTO {
  key: string;
  label: string;
  value: string;
  subtext: string;
  breakdown?: RiskBreakdownDTO | null;
}

export interface TrendPointDTO {
  date: string;
  shrimp?: number;
  monthlyImport?: number | null;
  priceIndex?: number | null;
}

export type EvidenceIconType = 'globe' | 'trending' | 'file' | 'map';
export type EvidenceIconColor = 'green' | 'red' | 'neutral';

export interface EvidenceItemDTO {
  iconType: EvidenceIconType;
  iconColor?: EvidenceIconColor;
  title: string;
  description: string;
  source: string;
  impact: string;
  date: string;
  url?: string;
  relevancyScore?: number;
}

export type RecommendationIconType = 'cart' | 'refresh' | 'clock' | 'alert';

export interface RecommendationDTO {
  iconType: RecommendationIconType;
  action: string;
  product: string;
  description: string;
  priority: string;
  savings: string;
  timeline: string;
}

export interface DashboardResponse {
  meta: {
    asOf: string | null;
    hasData: boolean;
    generatedAt: string;
    /** True when any section uses sample data (see placeholderSections). */
    usingPlaceholders?: boolean;
    /** e.g. database_unavailable | empty_months_shrimp when entire payload is placeholder. */
    placeholderReason?: string;
    placeholderSections?: string[];
    dbError?: string | null;
  };
  overview: OverviewMetricDTO[];
  products: DashboardProductRow[];
  trend: TrendPointDTO[];
  evidence: EvidenceItemDTO[];
  recommendations: RecommendationDTO[];
  anomalies?: Record<string, number>;
}
