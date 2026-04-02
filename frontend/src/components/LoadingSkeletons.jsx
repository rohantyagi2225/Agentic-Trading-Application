/**
 * Premium Loading Skeletons for Agentic Trading Platform
 * Glassmorphism effects with smooth animations
 */

import { PREMIUM_COLORS } from '../styles/premiumDesignSystem';

export function SkeletonCard({ className = '' }) {
  return (
    <div className={`rounded-3xl ${PREMIUM_COLORS.glass.medium} border border-zinc-800/50 p-6 ${className}`}>
      <div className="flex items-center gap-4 mb-4">
        <div className="skeleton h-10 w-10 rounded-xl"></div>
        <div className="flex-1">
          <div className="skeleton h-4 w-32 rounded mb-2"></div>
          <div className="skeleton h-3 w-20 rounded"></div>
        </div>
      </div>
      <div className="skeleton h-20 w-full rounded-lg mt-4"></div>
    </div>
  );
}

export function SkeletonMetric({ label, className = '' }) {
  return (
    <div className={`${PREMIUM_COLORS.glass.dark} rounded-2xl p-5 border border-zinc-800/50 ${className}`}>
      <div className="skeleton h-3 w-24 rounded mb-3"></div>
      <div className="skeleton h-8 w-32 rounded-lg"></div>
    </div>
  );
}

export function SkeletonChart({ height = 'h-64' }) {
  return (
    <div className={`${PREMIUM_COLORS.glass.medium} rounded-3xl border border-zinc-800/50 p-6`}>
      <div className="skeleton h-6 w-48 rounded mb-6"></div>
      <div className={`skeleton w-full ${height} rounded-2xl`}></div>
    </div>
  );
}

export function SkeletonTable({ rows = 5 }) {
  return (
    <div className={`${PREMIUM_COLORS.glass.medium} rounded-3xl border border-zinc-800/50 p-6`}>
      <div className="skeleton h-6 w-32 rounded mb-6"></div>
      <div className="space-y-3">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="flex items-center gap-4">
            <div className="skeleton h-10 w-10 rounded-lg"></div>
            <div className="flex-1 space-y-2">
              <div className="skeleton h-4 w-48 rounded"></div>
              <div className="skeleton h-3 w-32 rounded"></div>
            </div>
            <div className="skeleton h-8 w-24 rounded-lg"></div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function SkeletonDashboard() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      <SkeletonMetric />
      <SkeletonMetric />
      <SkeletonMetric />
      <SkeletonMetric />
    </div>
  );
}

export function SkeletonMarketCard() {
  return (
    <div className={`${PREMIUM_COLORS.glass.medium} rounded-3xl border border-zinc-800/50 p-6 hover:border-cyan-500/30 transition-all duration-300`}>
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="skeleton h-6 w-24 rounded mb-2"></div>
          <div className="skeleton h-4 w-32 rounded"></div>
        </div>
        <div className="skeleton h-8 w-20 rounded-lg"></div>
      </div>
      <div className="skeleton h-24 w-full rounded-2xl mb-4"></div>
      <div className="flex gap-2">
        <div className="skeleton h-6 w-16 rounded-full"></div>
        <div className="skeleton h-6 w-16 rounded-full"></div>
      </div>
    </div>
  );
}
