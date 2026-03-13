import { useState, useEffect, useCallback } from 'react';

export function usePolling(fetchFn, interval = 5000, immediate = true) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetch = useCallback(async () => {
    try {
      const result = await fetchFn();
      if (result !== null) setData(result);
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [fetchFn]);

  useEffect(() => {
    if (immediate) fetch();
    const timer = setInterval(fetch, interval);
    return () => clearInterval(timer);
  }, [fetch, interval, immediate]);

  return { data, loading, error, refetch: fetch };
}
