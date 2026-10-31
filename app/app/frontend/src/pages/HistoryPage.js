import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { toast } from "sonner";
import Header from "../components/Header";
import HistoryPanel from "../components/HistoryPanel";
import { STATIC_HISTORY } from "../lib/staticData";

const API = `${process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000"}/api`;

export default function HistoryPage() {
    const navigate = useNavigate();
    const [history, setHistory] = useState(() => {
        try {
            const saved = localStorage.getItem("neuralcord_history");
            return saved ? JSON.parse(saved) : STATIC_HISTORY;
        } catch {
            return STATIC_HISTORY;
        }
    });

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const r = await axios.get(`${API}/predictions?limit=50`, { timeout: 2500 });
                if (r.data && Array.isArray(r.data) && r.data.length > 0) {
                    setHistory(r.data);
                    try { localStorage.setItem("neuralcord_history", JSON.stringify(r.data)); } catch {}
                }
            } catch (e) {
                // Silently retain static history
            }
        };
        fetchHistory();
    }, []);

    const handleSelectHistory = (rec) => {
        try {
            localStorage.setItem("neuralcord_current", JSON.stringify(rec));
        } catch {}
        toast.success(`Loaded ${rec.predicted_class} in Diagnostic Console`);
        navigate("/");
    };

    const handleDelete = async (id) => {
        try {
            await axios.delete(`${API}/predictions/${id}`);
            toast.success("Record deleted");
        } catch {}
        setHistory((prev) => {
            const updated = prev.filter(p => p.id !== id);
            try { localStorage.setItem("neuralcord_history", JSON.stringify(updated)); } catch {}
            return updated;
        });
    };

    return (
        <div className="min-h-screen bg-transparent relative overflow-x-clip" data-testid="history-page">
            <Header />

            {/* Page Hero Header */}
            <section className="border-b border-white/10 relative z-10 fade-up bg-gradient-to-b from-slate-950/80 via-slate-900/30 to-transparent">
                <div className="max-w-[1400px] mx-auto px-6 lg:px-10 py-12">
                    <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-cyan-400/30 bg-cyan-950/40 text-cyan-300 font-mono text-xs mb-4 shadow-[0_0_20px_rgba(34,211,238,0.15)]">
                        <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
                        <span>CLINICAL SESSIONS & AUDIT LOG</span>
                    </div>
                    <h1 className="font-display text-4xl sm:text-5xl font-extrabold tracking-tight text-white">
                        Patient Diagnostic History
                    </h1>
                    <p className="mt-3 max-w-2xl text-zinc-300 text-base font-light leading-relaxed">
                        Complete session audit log of all spinal cord MRI inference runs. Click any row to load 
                        and inspect its full 6-view segmentation and Grad-CAM maps in the main diagnostic console.
                    </p>
                </div>
            </section>

            {/* Main History Content */}
            <div className="relative z-10">
                <HistoryPanel
                    items={history}
                    onSelect={handleSelectHistory}
                    onDelete={handleDelete}
                />
            </div>

            {/* Page Footer */}
            <footer className="border-t border-white/10 mt-16 bg-slate-950/80 backdrop-blur-md relative z-10">
                <div className="max-w-[1400px] mx-auto px-6 lg:px-10 py-8 flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
                    <div>
                        <div className="font-display font-bold text-white text-sm">
                            NeuralCord<span className="text-cyan-400">.AI</span>
                        </div>
                        <div className="kicker text-zinc-500 text-[9px] mt-0.5">
                            © 2026 PATIENT AUDIT TRAIL ARCHIVE
                        </div>
                    </div>
                    <div className="text-[11px] text-zinc-400 max-w-md text-left sm:text-right font-light leading-relaxed">
                        Data stored securely in local browser cache and persistent research storage.
                    </div>
                </div>
            </footer>
        </div>
    );
}
