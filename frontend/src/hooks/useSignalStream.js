import { useState, useEffect, useRef } from 'react';
import { SignalWebSocket } from '../services/websocket';

export function useSignalStream(symbol, maxMessages = 50) {
  const [messages, setMessages] = useState([]);
  const [status, setStatus] = useState('idle');
  const wsRef = useRef(null);

  useEffect(() => {
    if (!symbol) return;
    const ws = new SignalWebSocket(
      symbol,
      (msg) => setMessages((prev) => [{ ...msg, _ts: Date.now() }, ...prev].slice(0, maxMessages)),
      setStatus
    );
    wsRef.current = ws;
    ws.connect();
    return () => ws.disconnect();
  }, [symbol, maxMessages]);

  return { messages, status };
}
