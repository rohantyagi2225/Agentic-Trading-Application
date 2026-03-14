import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../services/api";

export default function VerifyEmail() {
  const [params] = useSearchParams();
  const token = params.get("token") ?? "";
  const [status, setStatus] = useState("Verifying your email...");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!token) {
      setError("Missing verification token");
      return;
    }
    api.verifyEmail(token)
      .then((result) => setStatus(result.message ?? "Email verified. You can now log in."))
      .catch((err) => setError(err?.message ?? "Verification failed"));
  }, [token]);

  return (
    <div className="min-h-[60vh] flex items-center justify-center p-4">
      <div className="w-full max-w-lg rounded-2xl border border-zinc-800 bg-zinc-900/70 p-8 text-center space-y-4">
        <div className="text-cyan-400 font-mono text-xs uppercase tracking-[0.3em]">Email Verification</div>
        {error ? (
          <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-red-300">{error}</div>
        ) : (
          <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-4 text-emerald-300">{status}</div>
        )}
        <Link to="/login" className="inline-flex rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-5 py-3 text-sm text-cyan-300 hover:bg-cyan-500/20">
          Go to Login
        </Link>
      </div>
    </div>
  );
}
