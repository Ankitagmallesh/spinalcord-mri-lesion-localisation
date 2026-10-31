import Header from "../components/Header";
import ArchitecturePanel from "../components/ArchitecturePanel";

export default function ArchitecturePage() {
    return (
        <div className="min-h-screen bg-transparent relative overflow-x-clip" data-testid="architecture-page">
            <Header />

            {/* Page Hero Header */}
            <section className="border-b border-white/10 relative z-10 fade-up bg-gradient-to-b from-slate-950/80 via-slate-900/30 to-transparent">
                <div className="max-w-[1400px] mx-auto px-6 lg:px-10 py-12">
                    <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-blue-400/30 bg-blue-950/40 text-blue-300 font-mono text-xs mb-4 shadow-[0_0_20px_rgba(59,130,246,0.15)]">
                        <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></span>
                        <span>NEURAL NETWORK ARCHITECTURES & BENCHMARKS</span>
                    </div>
                    <h1 className="font-display text-4xl sm:text-5xl font-extrabold tracking-tight text-white">
                        Comparative Model Architecture
                    </h1>
                    <p className="mt-3 max-w-2xl text-zinc-300 text-base font-light leading-relaxed">
                        In-depth comparative study of Standard U-Net, Attention Gated U-Net, and our proposed 
                        Dual-Task Residual Cross-Attention network (NeuroCord-ResAttnNet).
                    </p>
                </div>
            </section>

            {/* Main Architecture Content */}
            <div className="relative z-10">
                <ArchitecturePanel />
            </div>

            {/* Page Footer */}
            <footer className="border-t border-white/10 mt-16 bg-slate-950/80 backdrop-blur-md relative z-10">
                <div className="max-w-[1400px] mx-auto px-6 lg:px-10 py-8 flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
                    <div>
                        <div className="font-display font-bold text-white text-sm">
                            NeuralCord<span className="text-cyan-400">.AI</span>
                        </div>
                        <div className="kicker text-zinc-500 text-[9px] mt-0.5">
                            © 2026 ARCHITECTURE & MODEL BENCHMARK SUITE
                        </div>
                    </div>
                    <div className="text-[11px] text-zinc-400 max-w-md text-left sm:text-right font-light leading-relaxed">
                        Evaluated across 4-class multi-center spinal cord MRI benchmark dataset.
                    </div>
                </div>
            </footer>
        </div>
    );
}
