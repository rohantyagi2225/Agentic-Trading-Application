const WS_BASE = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

export class SignalWebSocket {
  constructor(symbol, onMessage, onStatusChange) {
    this.symbol = symbol;
    this.onMessage = onMessage;
    this.onStatusChange = onStatusChange;
    this.ws = null;
    this.reconnectTimer = null;
    this.attempts = 0;
    this.maxAttempts = 5;
    this.active = true;
  }

  connect() {
    if (!this.active) return;
    try {
      this.ws = new WebSocket(`${WS_BASE}/ws/signals/${this.symbol}`);
      this.onStatusChange?.('connecting');

      this.ws.onopen = () => {
        this.attempts = 0;
        this.onStatusChange?.('connected');
      };

      this.ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          this.onMessage?.(data);
        } catch {
          this.onMessage?.({ raw: e.data });
        }
      };

      this.ws.onerror = () => this.onStatusChange?.('error');

      this.ws.onclose = () => {
        if (!this.active) return;
        this.onStatusChange?.('disconnected');
        this.scheduleReconnect();
      };
    } catch (e) {
      this.onStatusChange?.('error');
      this.scheduleReconnect();
    }
  }

  scheduleReconnect() {
    if (this.attempts >= this.maxAttempts || !this.active) return;
    const delay = Math.min(1000 * 2 ** this.attempts, 30000);
    this.attempts++;
    this.reconnectTimer = setTimeout(() => this.connect(), delay);
  }

  disconnect() {
    this.active = false;
    clearTimeout(this.reconnectTimer);
    this.ws?.close();
  }
}
