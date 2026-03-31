import { useEffect, useMemo, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { api } from '../services/api';
import { searchMarkets } from '../data/marketCatalog';
import { PREMIUM_COLORS, ANIMATIONS, COMPONENTS, TYPOGRAPHY } from '../styles/premiumDesignSystem';

const NAV_LINKS = [
  { to: '/markets', label: 'Markets' },
  { to: '/learn', label: 'Learn' },
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/portfolio', label: 'Portfolio' },
  { to: '/profile', label: 'Profile' },
  { to: '/agents', label: 'AI Agents' },
];

export default function Navbar() {
  const { isAuthenticated, user, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [scrolled, setScrolled] = useState(false);

  // Detect scroll for glassmorphism effect
  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const isActive = (to) => location.pathname.startsWith(to);

  useEffect(() => {
    const timer = window.setTimeout(async () => {
      const trimmed = query.trim();
      if (!trimmed) {
        setSuggestions([]);
        return;
      }

      try {
        const response = await api.searchSymbols(trimmed);
        const remote = response?.results || [];
        const local = searchMarkets(trimmed).map((item) => ({
          symbol: item.symbol,
          name: item.name,
          sector: item.sector,
        }));
        setSuggestions(Array.from(new Map([...local, ...remote].map((item) => [item.symbol, item])).values()).slice(0, 6));
      } catch {
        setSuggestions(searchMarkets(trimmed).slice(0, 6));
      }
    }, 180);

    return () => window.clearTimeout(timer);
  }, [query]);

  const topIdentity = useMemo(() => user?.full_name || user?.email || 'Trader', [user]);

  const submitSearch = (event) => {
    event.preventDefault();
    const next = suggestions[0];
    if (next?.symbol) {
      navigate(`/markets/${next.symbol}`);
      setQuery('');
      setSuggestions([]);
    } else if (query.trim()) {
      navigate(`/markets?query=${encodeURIComponent(query.trim())}`);
      setSuggestions([]);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/');
    setUserMenuOpen(false);
  };

  const openAssistant = () => {
    window.dispatchEvent(new Event('agentic:assistant-open'));
    setMobileOpen(false);
  };

  return (
    <nav className={`sticky top-0 z-50 transition-all duration-500 ${
      scrolled 
        ? 'bg-black/80 backdrop-blur-2xl border-zinc-800/50 shadow-2xl shadow-black/50' 
        : 'bg-transparent border-transparent'
    } border-b`}>
      <div className="max-w-[1680px] mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex min-h-[72px] items-center gap-4">
          {/* Logo - Enhanced with glow effect */}
          <Link to="/" className="group flex items-center gap-3 shrink-0">
            <div className="relative flex h-10 w-10 items-center justify-center rounded-2xl border border-cyan-400/30 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 text-cyan-300 text-sm font-bold transition-all duration-300 group-hover:scale-110 group-hover:shadow-[0_0_30px_-5px_rgba(34,211,238,0.4)]">
              AGT
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-cyan-400/20 to-blue-400/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            </div>
            <div className="hidden md:block">
              <div className="text-base font-bold bg-clip-text text-transparent bg-gradient-to-r from-zinc-100 via-zinc-200 to-zinc-100">AgenticTrading</div>
              <div className="text-[10px] uppercase tracking-[0.25em] text-zinc-500">AI-Powered Trading Platform</div>
            </div>
          </Link>

          {/* Search Bar - Premium Design */}
          <form onSubmit={submitSearch} className="relative hidden lg:block flex-1 max-w-2xl">
            <div className="group relative">
              <div className="absolute left-4 top-1/2 -translate-y-1/2 text-zinc-500 group-focus-within:text-cyan-400 transition-colors duration-300">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search markets, companies, or symbols..."
                className="w-full rounded-2xl border border-zinc-800/80 bg-zinc-900/60 backdrop-blur-xl pl-12 pr-5 py-3.5 text-sm text-zinc-100 placeholder-zinc-600 outline-none transition-all duration-300 focus:border-cyan-500/50 focus:bg-zinc-900/80 focus:shadow-[0_0_30px_-8px_rgba(34,211,238,0.3)] hover:border-zinc-700"
              />
              <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
                <kbd className="hidden md:inline-flex h-6 items-center gap-1 rounded-md border border-zinc-700 bg-zinc-800/50 px-2 font-mono text-[10px] text-zinc-500">
                  <span className="text-xs">⌘</span>K
                </kbd>
              </div>
            </div>
            
            {/* Enhanced Suggestions Dropdown */}
            {!!suggestions.length && (
              <div className="absolute left-0 right-0 top-full z-50 mt-3 overflow-hidden rounded-2xl border border-zinc-800/80 bg-zinc-900/95 backdrop-blur-2xl shadow-2xl shadow-black/80 animate-in slide-in-from-top-2 duration-200">
                <div className="max-h-[400px] overflow-y-auto scrollbar-thin scrollbar-track-zinc-900 scrollbar-thumb-zinc-700">
                  {suggestions.map((item, idx) => (
                    <button
                      key={item.symbol}
                      type="button"
                      onClick={() => {
                        navigate(`/markets/${item.symbol}`);
                        setQuery('');
                        setSuggestions([]);
                      }}
                      className={`group flex w-full items-center justify-between px-5 py-4 text-left transition-all duration-200 ${
                        idx === 0 ? 'bg-gradient-to-r from-cyan-500/10 to-transparent' : ''
                      } hover:bg-zinc-800/60 border-b border-zinc-800/50 last:border-0`}
                    >
                      <div className="flex items-center gap-3">
                        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-500/20 text-cyan-400 font-mono text-sm font-bold border border-cyan-500/20 group-hover:scale-110 transition-transform duration-200">
                          {item.symbol}
                        </div>
                        <div>
                          <div className="font-semibold text-zinc-100 group-hover:text-white">{item.name}</div>
                          <div className="text-xs text-zinc-500 group-hover:text-zinc-400">{item.sector}</div>
                        </div>
                      </div>
                      <div className="text-[10px] uppercase tracking-widest text-zinc-600 group-hover:text-cyan-400 transition-colors">
                        →
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </form>

          {/* Navigation Links - Enhanced */}
          <div className="hidden xl:flex items-center gap-1">
            {NAV_LINKS.map((link) => (
              <Link
                key={link.to}
                to={link.to}
                className={`relative rounded-xl px-4 py-2.5 text-sm font-medium transition-all duration-300 group ${
                  isActive(link.to)
                    ? 'text-white'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                {isActive(link.to) && (
                  <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-cyan-500/20 to-blue-500/20 border border-cyan-500/30"></div>
                )}
                <span className="relative z-10 flex items-center gap-2">
                  {link.label}
                  <span className={`h-1.5 w-1.5 rounded-full ${
                    isActive(link.to) ? 'bg-cyan-400' : 'bg-transparent group-hover:bg-cyan-400/50'
                  } transition-all duration-300`}></span>
                </span>
              </Link>
            ))}
            <button
              type="button"
              onClick={openAssistant}
              className="rounded-xl px-3 py-2 text-sm text-zinc-500 transition-colors hover:bg-zinc-800/60 hover:text-zinc-200"
            >
              AI Assistant
            </button>
          </div>

          <div className="ml-auto flex items-center gap-2">
            {isAuthenticated ? (
              <div className="relative">
                <button
                  onClick={() => setUserMenuOpen((value) => !value)}
                  className="flex items-center gap-2 rounded-2xl border border-zinc-800 bg-zinc-900/70 px-3 py-2"
                >
                  <div className="flex h-7 w-7 items-center justify-center rounded-full border border-cyan-500/30 bg-cyan-500/10 text-[11px] font-bold text-cyan-300">
                    {topIdentity[0]?.toUpperCase()}
                  </div>
                  <div className="hidden sm:block text-left">
                    <div className="max-w-[10rem] truncate text-sm text-zinc-200">{topIdentity}</div>
                    <div className="text-[10px] font-mono text-zinc-500">${Number(user?.demo_balance || 0).toLocaleString('en-US')}</div>
                  </div>
                </button>

                {userMenuOpen && (
                  <div className="absolute right-0 top-full z-50 mt-3 w-64 overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900 shadow-2xl shadow-black/40">
                    <div className="border-b border-zinc-800 px-4 py-4">
                      <div className="text-sm text-zinc-100">{topIdentity}</div>
                      <div className="mt-1 text-xs text-zinc-500">{user?.email}</div>
                    </div>
                    <div className="p-2">
                      {['/dashboard', '/portfolio', '/profile', '/agents'].map((to) => (
                        <Link key={to} to={to} onClick={() => setUserMenuOpen(false)} className="block rounded-xl px-3 py-2 text-sm text-zinc-300 hover:bg-zinc-800">
                          {to.replace('/', '').replace(/^\w/, (char) => char.toUpperCase())}
                        </Link>
                      ))}
                      <button onClick={handleLogout} className="mt-1 w-full rounded-xl px-3 py-2 text-left text-sm text-red-400 hover:bg-red-500/10">
                        Sign Out
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <>
                <Link to="/login" className="hidden sm:block text-sm text-zinc-400 hover:text-zinc-200">Login</Link>
                <Link to="/signup" className="btn-primary">Get Started</Link>
              </>
            )}

            <button
              onClick={() => setMobileOpen((value) => !value)}
              className="xl:hidden rounded-xl p-2 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={mobileOpen ? 'M6 18L18 6M6 6l12 12' : 'M4 6h16M4 12h16M4 18h16'} />
              </svg>
            </button>
          </div>
        </div>

        {mobileOpen && (
          <div className="border-t border-zinc-800/70 py-4 xl:hidden">
            <form onSubmit={submitSearch} className="mb-3">
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search tickers or companies..."
                className="w-full rounded-xl border border-zinc-800 bg-zinc-900 px-4 py-3 text-sm text-zinc-100 outline-none focus:border-cyan-500"
              />
            </form>
            <div className="grid gap-1">
              {NAV_LINKS.map((link) => (
                <Link
                  key={link.to}
                  to={link.to}
                  onClick={() => setMobileOpen(false)}
                  className={`rounded-xl px-3 py-2 text-sm ${
                    isActive(link.to) ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200'
                  }`}
                >
                  {link.label}
                </Link>
              ))}
              <button
                type="button"
                onClick={openAssistant}
                className="rounded-xl px-3 py-2 text-left text-sm text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
              >
                AI Assistant
              </button>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}
