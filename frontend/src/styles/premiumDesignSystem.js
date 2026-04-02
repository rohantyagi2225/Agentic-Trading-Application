/**
 * Premium Design System for Agentic Trading Platform
 * Sigma-level quality with advanced animations, glassmorphism, and micro-interactions
 */

// ============================================================================
// PREMIUM COLOR PALETTE
// ============================================================================

export const PREMIUM_COLORS = {
  // Gradients
  gradients: {
    primary: 'bg-gradient-to-r from-blue-600 via-cyan-500 to-emerald-500',
    secondary: 'bg-gradient-to-br from-zinc-900 via-zinc-800 to-zinc-900',
    accent: 'bg-gradient-to-r from-violet-600 via-purple-500 to-fuchsia-500',
    success: 'bg-gradient-to-r from-emerald-500 via-teal-400 to-cyan-400',
    danger: 'bg-gradient-to-r from-red-600 via-rose-500 to-pink-500',
    warm: 'bg-gradient-to-r from-amber-500 via-orange-400 to-yellow-400',
  },
  
  // Glass morphism effects
  glass: {
    light: 'bg-white/5 backdrop-blur-xl border border-white/10',
    dark: 'bg-black/40 backdrop-blur-xl border border-white/5',
    medium: 'bg-zinc-900/60 backdrop-blur-xl border border-zinc-700/50',
    strong: 'bg-zinc-950/80 backdrop-blur-2xl border border-zinc-800/60',
  },
  
  // Glow effects
  glow: {
    primary: 'shadow-[0_0_40px_-10px_rgba(59,130,246,0.5)]',
    success: 'shadow-[0_0_40px_-10px_rgba(16,185,129,0.5)]',
    danger: 'shadow-[0_0_40px_-10px_rgba(239,68,68,0.5)]',
    warm: 'shadow-[0_0_40px_-10px_rgba(245,158,11,0.5)]',
  },
  
  // Premium backgrounds
  backgrounds: {
    hero: 'bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-zinc-900 via-zinc-950 to-black',
    card: 'bg-gradient-to-br from-zinc-900/80 via-zinc-900/50 to-zinc-900/80',
    panel: 'bg-gradient-to-b from-zinc-900/60 to-zinc-950/60',
    overlay: 'bg-black/60 backdrop-blur-2xl',
  },
};

// ============================================================================
// ADVANCED ANIMATIONS
// ============================================================================

export const ANIMATIONS = {
  // Transitions
  transition: {
    fast: 'transition-all duration-200 ease-out',
    normal: 'transition-all duration-300 ease-out',
    slow: 'transition-all duration-500 ease-out',
    spring: 'transition-all duration-500 cubic-bezier(0.34, 1.56, 0.64, 1)',
  },
  
  // Hover effects
  hover: {
    lift: 'hover:-translate-y-1 hover:shadow-2xl',
    glow: 'hover:shadow-[0_0_30px_-5px_rgba(59,130,246,0.4)]',
    scale: 'hover:scale-105',
    shimmer: 'hover:before:opacity-100',
  },
  
  // Keyframe-based animations (add to tailwind.config.js)
  keyframes: {
    'fade-in': 'fadeIn 0.5s ease-out',
    'slide-up': 'slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1)',
    'slide-down': 'slideDown 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
    'scale-in': 'scaleIn 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)',
    'shimmer': 'shimmer 2s linear infinite',
    'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
    'float': 'float 3s ease-in-out infinite',
  },
};

// ============================================================================
// TYPOGRAPHY SYSTEM
// ============================================================================

export const TYPOGRAPHY = {
  // Font families
  fonts: {
    sans: 'font-sans',
    mono: 'font-mono',
    display: 'font-[Inter,_system-ui,_sans-serif]',
  },
  
  // Heading styles
  headings: {
    hero: 'text-5xl md:text-7xl font-bold tracking-tight',
    h1: 'text-4xl md:text-5xl font-bold tracking-tight',
    h2: 'text-3xl md:text-4xl font-semibold tracking-tight',
    h3: 'text-2xl md:text-3xl font-semibold tracking-tight',
    h4: 'text-xl md:text-2xl font-medium',
  },
  
  // Body styles
  body: {
    large: 'text-lg leading-relaxed',
    base: 'text-base leading-relaxed',
    small: 'text-sm leading-relaxed',
    tiny: 'text-xs leading-relaxed',
  },
  
  // Special text effects
  effects: {
    gradient: 'bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-cyan-400 to-emerald-400',
    gradientWarm: 'bg-clip-text text-transparent bg-gradient-to-r from-amber-400 via-orange-400 to-yellow-400',
    gradientPurple: 'bg-clip-text text-transparent bg-gradient-to-r from-violet-400 via-purple-400 to-fuchsia-400',
    glow: 'drop-shadow-[0_0_10px_rgba(59,130,246,0.5)]',
  },
};

// ============================================================================
// COMPONENT STYLES
// ============================================================================

export const COMPONENTS = {
  // Buttons
  button: {
    primary: `
      ${PREMIUM_COLORS.gradients.primary} 
      text-white font-semibold px-6 py-3 rounded-xl
      ${ANIMATIONS.transition.normal}
      ${ANIMATIONS.hover.lift} ${ANIMATIONS.hover.glow}
      hover:brightness-110 active:scale-95
      shadow-lg shadow-blue-500/30
    `,
    secondary: `
      ${PREMIUM_COLORS.glass.light}
      text-white font-semibold px-6 py-3 rounded-xl
      ${ANIMATIONS.transition.normal}
      ${ANIMATIONS.hover.lift}
      hover:bg-white/10 active:scale-95
    `,
    ghost: `
      text-zinc-400 hover:text-white font-medium px-4 py-2 rounded-lg
      ${ANIMATIONS.transition.fast}
      hover:bg-white/5 active:scale-95
    `,
    icon: `
      p-3 rounded-xl ${PREMIUM_COLORS.glass.light}
      ${ANIMATIONS.transition.fast}
      hover:bg-white/10 hover:scale-110 active:scale-95
    `,
  },
  
  // Cards
  card: {
    base: `
      ${PREMIUM_COLORS.backgrounds.card}
      ${PREMIUM_COLORS.glass.medium}
      rounded-2xl p-6
      ${ANIMATIONS.transition.normal}
      ${ANIMATIONS.hover.lift}
    `,
    interactive: `
      ${PREMIUM_COLORS.backgrounds.card}
      ${PREMIUM_COLORS.glass.medium}
      rounded-2xl p-6 cursor-pointer
      ${ANIMATIONS.transition.normal}
      ${ANIMATIONS.hover.lift} ${ANIMATIONS.hover.glow}
      hover:border-blue-500/50
    `,
    premium: `
      ${PREMIUM_COLORS.backgrounds.card}
      ${PREMIUM_COLORS.glass.light}
      rounded-3xl p-8
      ${ANIMATIONS.transition.slow}
      ${ANIMATIONS.hover.lift}
      relative overflow-hidden
      before:absolute before:inset-0 before:bg-gradient-to-r before:from-blue-500/10 before:via-transparent before:to-emerald-500/10
      before:opacity-0 before:transition-opacity before:duration-500
      hover:before:opacity-100
    `,
  },
  
  // Inputs
  input: {
    base: `
      ${PREMIUM_COLORS.glass.dark}
      text-white placeholder-zinc-500
      px-4 py-3 rounded-xl
      ${ANIMATIONS.transition.fast}
      focus:bg-white/10 focus:border-blue-500/50 focus:outline-none focus:ring-2 focus:ring-blue-500/20
      disabled:opacity-50 disabled:cursor-not-allowed
    `,
    withIcon: `
      ${PREMIUM_COLORS.glass.dark}
      text-white placeholder-zinc-500
      pl-11 pr-4 py-3 rounded-xl
      ${ANIMATIONS.transition.fast}
      focus:bg-white/10 focus:border-blue-500/50 focus:outline-none focus:ring-2 focus:ring-blue-500/20
    `,
  },
  
  // Badges
  badge: {
    success: `
      bg-emerald-500/20 text-emerald-400 border border-emerald-500/30
      px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider
    `,
    danger: `
      bg-red-500/20 text-red-400 border border-red-500/30
      px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider
    `,
    warning: `
      bg-amber-500/20 text-amber-400 border border-amber-500/30
      px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider
    `,
    info: `
      bg-blue-500/20 text-blue-400 border border-blue-500/30
      px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider
    `,
    gradient: `
      bg-gradient-to-r from-blue-500/20 via-cyan-500/20 to-emerald-500/20
      text-white border border-white/20
      px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider
      backdrop-blur-sm
    `,
  },
};

// ============================================================================
// LAYOUT UTILITIES
// ============================================================================

export const LAYOUT = {
  // Container
  container: {
    base: 'max-w-7xl mx-auto px-4 sm:px-6 lg:px-8',
    narrow: 'max-w-5xl mx-auto px-4 sm:px-6 lg:px-8',
    wide: 'max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8',
  },
  
  // Grid patterns
  grid: {
    auto: 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6',
    dashboard: 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6',
    featured: 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8',
  },
  
  // Flex patterns
  flex: {
    center: 'flex items-center justify-center',
    between: 'flex items-center justify-between',
    start: 'flex items-center justify-start',
    end: 'flex items-center justify-end',
    col: 'flex flex-col',
    responsive: 'flex flex-col md:flex-row',
  },
};

// ============================================================================
// SPECIAL EFFECTS
// ============================================================================

export const EFFECTS = {
  // Scrollbar styling (add to global CSS)
  scrollbar: `
    scrollbar-thin scrollbar-track-zinc-900 scrollbar-thumb-zinc-700
    scrollbar-thumb-rounded-full scrollbar-hover:thumb-zinc-600
  `,
  
  // Selection color
  selection: '::selection:bg-blue-500/30 ::selection:text-white',
  
  // Smooth scrolling
  smoothScroll: 'scroll-smooth',
  
  // Anti-aliasing
  antialias: 'antialiased',
};

// ============================================================================
// EXPORTED THEME OBJECT
// ============================================================================

export const PREMIUM_THEME = {
  colors: PREMIUM_COLORS,
  animations: ANIMATIONS,
  typography: TYPOGRAPHY,
  components: COMPONENTS,
  layout: LAYOUT,
  effects: EFFECTS,
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

export function cn(...classes) {
  return classes.filter(Boolean).join(' ');
}

export function createGradient(from, via, to) {
  return `bg-gradient-to-r from-${from} via-${via} to-${to}`;
}

export function createGlass(intensity = 'medium') {
  return PREMIUM_COLORS.glass[intensity] || PREMIUM_COLORS.glass.medium;
}
