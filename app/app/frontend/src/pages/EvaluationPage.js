import { useState, useEffect } from "react";
import axios from "axios";
import Header from "../components/Header";
import MetricsPanel from "../components/MetricsPanel";
import { STATIC_METRICS } from "../lib/staticData";

const API = `${process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000"}/api`;

export default function EvaluationPage() {
    const [metrics, setMetrics] = useState(() => {
        try {
            const saved = localStorage.getItem("neuralcord_metrics");
            return saved ? JSON.parse(saved) : STATIC_METRICS;
        } catch {
            return STATIC_METRICS;
        }
    });

    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                const r = await axios.get(`${API}/metrics`, { timeout: 2500 });
                if (r.data && r.data.classes) {
                    setMetrics(r.data);
                }
            } catch (e) {
                // Silently retain static metrics
            }
        };
        fetchMetrics();
    }, []);

    return (
        <div className="min-h-screen bg-transparent relative overflow-x-clip" data-testid="evaluation-page">
            <Header />

            {/* Page Hero Header */}
            <section className="border-b border-white/10 relative z-10 fade-up bg-gradient-to-b from-slate-950/80 via-slate-900/30 to-transparent">
                <div className="max-w-[1400px] mx-auto px-6 lg:px-10 py-12">
                    <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-emerald-400/30 bg-emerald-950/40 text-emerald-300 font-mono text-xs mb-4 shadow-[0_0_20px_rgba(16,185,129,0.15)]">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                        <span>PERFORMANCE METRICS & CONFUSION MATRIX</span>
                    </div>
                    <h1 className="font-display text-4xl sm:text-5xl font-extrabold tracking-tight text-white">
                        Quantitative Model Evaluation
                    </h1>
                    <p className="mt-3 max-w-2xl text-zinc-300 text-base font-light leading-relaxed">
                        Comprehensive evaluation metrics across all test splits and active clinical predictions. 
                        Inspect per-class F1, Dice, IoU, and cross-class classification distributions.
                    </p>
                </div>
            </section>

            {/* Main Metrics Content */}
            <div className="relative z-10">
                <MetricsPanel metrics={metrics} />
            </div>

            {/* Page Footer */}
            <footer className="border-t border-white/10 mt-16 bg-slate-950/80 backdrop-blur-md relative z-10">
                <div className="max-w-[1400px] mx-auto px-6 lg:px-10 py-8 flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
                    <div>
                        <div className="font-display font-bold text-white text-sm">
                            NeuralCord<span className="text-cyan-400">.AI</span>
                        </div>
                        <div className="kicker text-zinc-500 text-[9px] mt-0.5">
                            © 2026 QUANTITATIVE EVALUATION SUITE
                        </div>
                    </div>
                    <div className="text-[11px] text-zinc-400 max-w-md text-left sm:text-right font-light leading-relaxed">
                        Evaluated using standard Dice Similarity Coefficient (DSC) and Jaccard Intersection over Union (IoU).
                    </div>
                </div>
            </footer>
        </div>
    );
}
