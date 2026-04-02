import { useEffect, useRef, useState } from 'react';

/**
 * TradingView Advanced Chart Widget
 * 
 * Features:
 * - Full technical analysis tools
 * - 100+ indicators and studies
 * - Drawing tools (trend lines, Fibonacci, etc.)
 * - Multiple chart types (Candlestick, Heikin Ashi, Bars, Line, Area)
 * - Timeframe selection (1m to 1M)
 * - Compare symbols functionality
 * - Pine Script strategy support
 * - Real-time data updates
 * - Customizable themes
 */
export default function TradingViewChart({ 
  symbol = 'AAPL', 
  theme = 'dark',
  interval = 'D',
  height = 600,
  showToolbar = true,
  showSettings = true,
  hideSideToolbar = false,
  allowSymbolChange = true,
  enablePublishing = false,
  saveImage = true,
}) {
  const containerRef = useRef(null);
  const widgetRef = useRef(null);
  const [loaded, setLoaded] = useState(false);
  const [timedOut, setTimedOut] = useState(false);

  useEffect(() => {
    // Clean up previous widget if exists
    if (containerRef.current) {
      containerRef.current.innerHTML = '';
    }
    setLoaded(false);
    setTimedOut(false);

    // TradingView widget configuration
    const config = {
      // Container & Theme
      container_id: "tradingview_chart",
      autosize: false,
      width: "100%",
      height: height,
      theme: theme,
      
      // Symbol & Interval
      symbol: symbol.toUpperCase(),
      interval: interval,
      
      // Chart Type & Style
      timezone: "America/New_York",
      style: showToolbar ? "1" : "1", // 1=Candles, 2=Line, 3=Bars, 8=Heikin Ashi, 9=Hollow Candles
      locale: "en",
      
      // Toolbar Configuration
      toolbar_bg: theme === 'dark' ? "#1a1a1a" : "#f1f3f6",
      withdateranges: true,
      hide_side_toolbar: hideSideToolbar,
      details: true,
      hotlist: true,
      calendar: true,
      
      // Features Toggle
      allow_symbol_change: allowSymbolChange,
      save_image: saveImage,
      enable_publishing: enablePublishing,
      
      // Studies & Indicators (pre-loaded)
      studies: [
        'Volume@tv-basicstudies',
        'RSI@tv-basicstudies',
        'MACD@tv-basicstudies',
        'Moving Average@tv-basicstudies',
        'Bollinger Bands@tv-basicstudies',
      ],
      
      // Customization
      overrides: {
        "paneProperties.background": theme === 'dark' ? "#0f172a" : "#ffffff",
        "paneProperties.vertGridProperties.color": theme === 'dark' ? "#1e293b" : "#e2e8f0",
        "paneProperties.horzGridProperties.color": theme === 'dark' ? "#1e293b" : "#e2e8f0",
        "scalesProperties.textColor": theme === 'dark' ? "#94a3b8" : "#64748b",
        "mainSeriesProperties.candleStyle.upColor": "#10b981",
        "mainSeriesProperties.candleStyle.downColor": "#ef4444",
        "mainSeriesProperties.candleStyle.drawBorder": true,
        "mainSeriesProperties.candleStyle.drawWick": true,
      },
      
      // Loading screen
      loading_screen: {
        backgroundColor: theme === 'dark' ? "#0f172a" : "#ffffff",
        foregroundColor: "#3b82f6",
      },
      
      // Disabled features (for cleaner UI)
      disabled_features: [
        'use_localstorage_for_settings',
        'header_widget',
        'symbol_info',
        'display_market_status',
        'go_to_date',
        'context_menus',
      ],
      
      // Enabled features
      enabled_features: [
        'left_toolbar',
        'timeframes',
        'chart_property_page_style',
        'show_interval_dialog_on_key_press',
        'chart_property_page_trading_settings',
        'datasource_copypaste_as_table',
      ],
      
      // Additional customizations
      custom_css_url: "",
      debug_mode: false,
    };

    // Create script element for TradingView widget
    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
    script.async = true;
    script.type = 'text/javascript';
    script.onload = () => setLoaded(true);
    
    // Serialize config to JSON string
    script.innerHTML = JSON.stringify(config, null, 2);
    
    // Append to container
    if (containerRef.current) {
      containerRef.current.appendChild(script);
      widgetRef.current = script;
    }

    const timeoutId = window.setTimeout(() => {
      setTimedOut(true);
    }, 10000);

    const observer = new MutationObserver(() => {
      const iframe = containerRef.current?.querySelector('iframe');
      if (iframe) {
        setLoaded(true);
        window.clearTimeout(timeoutId);
        observer.disconnect();
      }
    });

    if (containerRef.current) {
      observer.observe(containerRef.current, { childList: true, subtree: true });
    }

    // Cleanup on unmount
    return () => {
      window.clearTimeout(timeoutId);
      observer.disconnect();
      if (widgetRef.current && containerRef.current) {
        containerRef.current.removeChild(widgetRef.current);
      }
    };
  }, [symbol, theme, interval, height, showToolbar, hideSideToolbar, allowSymbolChange, enablePublishing, saveImage]);

  return (
    <div className="relative w-full">
      {/* TradingView Widget Container */}
      <div
        ref={containerRef}
        id="tradingview_chart"
        className="tradingview-widget-container relative rounded-xl overflow-hidden border border-zinc-800"
        style={{ height: `${height}px`, maxHeight: `${height}px` }}
      >
        {!loaded && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-950 z-10 pointer-events-none">
            <div className="flex flex-col items-center gap-3">
              <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
              <p className="text-xs text-zinc-500 font-mono">Loading TradingView Chart...</p>
            </div>
          </div>
        )}
        {timedOut && !loaded && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-zinc-950/95 z-20">
            <div className="text-xs font-mono text-amber-300">TradingView is taking too long to load.</div>
            <button
              onClick={() => {
                localStorage.setItem('chartType', 'legacy');
                window.dispatchEvent(new CustomEvent('chartTypeChange', { detail: { chartType: 'legacy' } }));
              }}
              className="text-[10px] font-mono text-cyan-400 hover:text-cyan-300"
            >
              Switch to Simple Candles
            </button>
          </div>
        )}
      </div>
      
      {/* Feature Info Badge */}
      <div className="mt-3 flex items-center justify-between px-2">
        <div className="flex items-center gap-2 text-[10px] font-mono text-zinc-500">
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
            Live Data
          </span>
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500"></span>
            100+ Indicators
          </span>
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-purple-500"></span>
            Drawing Tools
          </span>
        </div>
        
        <button
          onClick={() => {
            // Open in new window for full-screen experience
            window.open(`https://www.tradingview.com/chart/?symbol=${symbol}`, '_blank');
          }}
          className="text-[10px] font-mono text-blue-400 hover:text-blue-300 transition-colors"
        >
          Open Full Screen →
        </button>
      </div>
    </div>
  );
}
