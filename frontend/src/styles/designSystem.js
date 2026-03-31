/**
 * Professional Design System for Agentic Trading Platform
 * Consistent spacing, typography, colors, and components
 */

// ============================================================================
// COLOR PALETTE (Tailwind classes)
// ============================================================================

// Backgrounds
export const COLORS = {
  // Primary backgrounds
  bgPrimary: 'bg-zinc-950',        // Main background
  bgSecondary: 'bg-zinc-900',      // Cards, panels
  bgTertiary: 'bg-zinc-800',       // Hover states
  
  // Accent colors
  accent: {
    primary: 'text-blue-400',      // Primary actions
    success: 'text-emerald-400',   // Positive actions
    danger: 'text-red-400',        // Negative actions
    warning: 'text-amber-400',     // Warnings
    info: 'text-cyan-400',         // Information
  },
  
  // Text hierarchy
  text: {
    primary: 'text-zinc-100',
    secondary: 'text-zinc-400',
    tertiary: 'text-zinc-500',
    muted: 'text-zinc-600',
  },
  
  // Borders
  border: {
    default: 'border-zinc-800',
    subtle: 'border-zinc-700',
    highlight: 'border-zinc-600',
  }
};

// ============================================================================
// SPACING SYSTEM (in Tailwind units)
// ============================================================================

export const SPACING = {
  xs: 'p-2',      // 0.5rem - Tight spacing
  sm: 'p-3',      // 0.75rem - Compact spacing
  md: 'p-4',      // 1rem - Default spacing
  lg: 'p-6',      // 1.5rem - Generous spacing
  xl: 'p-8',      // 2rem - Section spacing
};

// ============================================================================
// TYPOGRAPHY
// ============================================================================

export const TYPOGRAPHY = {
  // Font families
  fontMain: 'font-sans',
  fontMono: 'font-mono',
  
  // Display headings (hero sections)
  display: 'text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight',
  
  // Page headings
  h1: 'text-3xl md:text-4xl font-bold',
  h2: 'text-2xl md:text-3xl font-semibold',
  h3: 'text-xl md:text-2xl font-medium',
  h4: 'text-lg md:text-xl font-medium',
  
  // Body text
  bodyLg: 'text-lg leading-relaxed',
  body: 'text-base leading-relaxed',
  bodySm: 'text-sm leading-normal',
  
  // Labels and captions
  label: 'text-xs font-medium uppercase tracking-wider',
  caption: 'text-xs text-zinc-500',
  
  // Data/Numbers
  dataLg: 'text-2xl font-mono font-semibold',
  data: 'text-lg font-mono',
  dataSm: 'text-sm font-mono',
};

// ============================================================================
// COMPONENT STYLES
// ============================================================================

// Card base style
export const cardBase = `
  ${COLORS.bgSecondary} 
  rounded-xl 
  border ${COLORS.border.default} 
  shadow-lg 
  overflow-hidden
`;

// Button variants
export const buttons = {
  primary: `
    bg-blue-600 
    hover:bg-blue-700 
    text-white 
    font-medium 
    px-4 py-2 
    rounded-lg 
    transition-colors
    duration-200
  `,
  
  secondary: `
    bg-zinc-800 
    hover:bg-zinc-700 
    text-zinc-100 
    font-medium 
    px-4 py-2 
    rounded-lg 
    transition-colors
    duration-200
  `,
  
  ghost: `
    bg-transparent 
    hover:bg-zinc-800 
    text-zinc-400 
    hover:text-zinc-100 
    font-medium 
    px-3 py-1.5 
    rounded-md 
    transition-colors
    duration-200
  `,
  
  danger: `
    bg-red-600 
    hover:bg-red-700 
    text-white 
    font-medium 
    px-4 py-2 
    rounded-lg 
    transition-colors
    duration-200
  `,
  
  success: `
    bg-emerald-600 
    hover:bg-emerald-700 
    text-white 
    font-medium 
    px-4 py-2 
    rounded-lg 
    transition-colors
    duration-200
  `,
};

// Input styles
export const inputs = {
  text: `
    w-full
    bg-zinc-900 
    border ${COLORS.border.subtle}
    focus:border-blue-500
    text-zinc-100
    placeholder-zinc-500
    px-4 
    py-2.5 
    rounded-lg
    outline-none
    transition-colors
    duration-200
  `,
};

// Badge styles
export const badges = {
  neutral: `
    inline-flex items-center 
    px-2.5 py-0.5 
    rounded-full 
    text-xs font-medium
    bg-zinc-800 
    text-zinc-400
  `,
  
  success: `
    inline-flex items-center 
    px-2.5 py-0.5 
    rounded-full 
    text-xs font-medium
    bg-emerald-900/30 
    text-emerald-400
  `,
  
  danger: `
    inline-flex items-center 
    px-2.5 py-0.5 
    rounded-full 
    text-xs font-medium
    bg-red-900/30 
    text-red-400
  `,
  
  warning: `
    inline-flex items-center 
    px-2.5 py-0.5 
    rounded-full 
    text-xs font-medium
    bg-amber-900/30 
    text-amber-400
  `,
  
  info: `
    inline-flex items-center 
    px-2.5 py-0.5 
    rounded-full 
    text-xs font-medium
    bg-cyan-900/30 
    text-cyan-400
  `,
};

// ============================================================================
// LAYOUT UTILITIES
// ============================================================================

export const layouts = {
  container: 'max-w-[1820px] mx-auto px-4 sm:px-6 lg:px-8',
  section: 'py-6 lg:py-8',
  grid: 'grid gap-4 md:gap-6',
  flex: 'flex gap-4',
  flexCenter: 'flex items-center justify-center',
  flexBetween: 'flex items-center justify-between',
};

// ============================================================================
// ANIMATION UTILITIES
// ============================================================================

export const animations = {
  fadeIn: 'animate-fadeIn',
  slideUp: 'animate-slideUp',
  pulse: 'animate-pulse',
  spin: 'animate-spin',
};

// ============================================================================
// RESPONSIVE BREAKPOINTS (for reference)
// ============================================================================

// sm: 640px
// md: 768px
// lg: 1024px
// xl: 1280px
// 2xl: 1536px
