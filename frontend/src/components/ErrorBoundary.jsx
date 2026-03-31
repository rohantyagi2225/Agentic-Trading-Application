import React, { Component } from 'react';
import { Link } from 'react-router-dom';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('❌ ErrorBoundary caught an error:', error);
    console.error('Component Stack:', errorInfo?.componentStack);
    this.setState({ error, errorInfo });
    
    // Log to error reporting service (optional)
    // logErrorToService(error, errorInfo);
  }

  handleReload = () => {
    window.location.reload();
  };

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-zinc-950 flex items-center justify-center p-4">
          <div className="max-w-md w-full bg-zinc-900 rounded-lg border border-zinc-800 p-6 text-center">
            <div className="mb-4">
              <svg 
                className="w-16 h-16 mx-auto text-red-500" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" 
                />
              </svg>
            </div>
            
            <h1 className="text-2xl font-bold text-zinc-100 mb-2">
              Oops! Something went wrong
            </h1>
            
            <p className="text-zinc-400 mb-6">
              We encountered an unexpected error. Don't worry, your data is safe.
            </p>
            
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <div className="mb-6 p-4 bg-zinc-950 rounded border border-zinc-800 text-left overflow-auto max-h-96">
                <div className="text-xs font-mono text-red-400 mb-2">
                  <strong>Error:</strong> {this.state.error.toString()}
                </div>
                {this.state.errorInfo && (
                  <div className="text-xs font-mono text-zinc-500">
                    <strong>Component Stack:</strong>
                    <pre className="mt-1 whitespace-pre-wrap break-words">
                      {this.state.errorInfo.componentStack}
                    </pre>
                  </div>
                )}
              </div>
            )}
            
            <div className="flex gap-3 justify-center">
              <button
                onClick={this.handleReload}
                className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-100 rounded transition"
              >
                🔄 Reload Page
              </button>
              
              <button
                onClick={this.handleReset}
                className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-100 rounded transition"
              >
                Try Again
              </button>
              
              <Link
                to="/"
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition"
              >
                Go Home
              </Link>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
