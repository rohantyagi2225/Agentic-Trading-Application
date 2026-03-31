import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { safeApi } from '../utils/safeApi';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);

  const fetchMe = useCallback(async (t) => {
    if (!t) { setLoading(false); return; }
    try {
      const result = await safeApi.getMe(t);
      if (result.success) {
        setUser(result.data);
      } else {
        localStorage.removeItem('token');
        setToken(null);
        setUser(null);
      }
    } catch {
      localStorage.removeItem('token');
      setToken(null);
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchMe(token); }, [fetchMe, token]);

  const login = useCallback(async (email, password) => {
    const result = await safeApi.login(email, password);
    if (!result.success) {
      throw new Error(result.error || 'Login failed');
    }
    localStorage.setItem('token', result.data.access_token);
    setToken(result.data.access_token);
    setUser(result.data.user);
    return result.data;
  }, []);

  const register = useCallback(async (email, password, full_name) => {
    const result = await safeApi.register(email, password, full_name);
    if (!result.success) {
      throw new Error(result.error || 'Registration failed');
    }
    return result.data;
  }, []);

  const logout = useCallback(() => {
    safeApi.logout().catch(() => {});
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout, isAuthenticated: !!user }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
};
