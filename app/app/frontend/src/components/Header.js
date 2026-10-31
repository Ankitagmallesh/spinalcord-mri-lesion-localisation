import { Link, NavLink } from "react-router-dom";
import { Activity, Sparkles, Shield, Cpu, Layers } from "lucide-react";

export default function Header() {
    const links = [
        { label: "Diagnostic Console", to: "/", testId: "nav-viewer" },
        { label: "Model Architecture", to: "/architecture", testId: "nav-arch" },
        { label: "Training Curves", to: "/training", testId: "nav-curves" },
        { label: "Evaluation Metrics", to: "/metrics", testId: "nav-metrics" },
        { label: "Patient History", to: "/history", testId: "nav-history" },
    ];

    return (
        <>
            <header className="fixed top-0 left-0 right-0 z-50 border-b border-white/10 bg-slate-950/90 backdrop-blur-2xl transition-all shadow-[0_4px_30px_rgba(0,0,0,0.5)]" data-testid="header">
                <div className="mx-auto flex h-16 max-w-[1400px] items-center justify-between px-6 lg:px-10">
                    
                    {/* Logo and Brand */}
                    <Link to="/" className="flex items-center gap-3.5 group cursor-pointer">
                        <div className="relative flex h-10 w-10 items-center justify-center rounded-xl border border-blue-400/40 bg-gradient-to-br from-blue-600 via-indigo-600 to-cyan-400 shadow-[0_0_20px_rgba(59,130,246,0.45)] transition-transform duration-300 group-hover:scale-105">
                            <Activity className="h-5 w-5 text-white animate-pulse" strokeWidth={2.4} />
                            <div className="absolute -inset-0.5 rounded-xl bg-blue-500 opacity-20 blur-sm group-hover:opacity-40 transition-opacity" />
                        </div>
                        <div>
                            <div className="font-display text-lg font-bold leading-none tracking-tight text-white flex items-center gap-1.5">
                                NeuralCord<span className="text-cyan-400">.AI</span>
                                <span className="px-1.5 py-0.5 rounded-md text-[9px] font-mono bg-blue-500/20 text-blue-300 border border-blue-400/30">v2.4</span>
                            </div>
                            <div className="kicker mt-1 text-[9px] text-zinc-400 flex items-center gap-1.5">
                                <span>SPINAL MRI DIAGNOSTICS</span>
                                <span className="text-zinc-600">·</span>
                                <span className="text-cyan-400/90 font-mono">U-NET + XAI</span>
                            </div>
                        </div>
                    </Link>

                    {/* Navigation Links with Active Highlighting */}
                    <nav className="hidden items-center gap-1.5 md:flex">
                        {links.map((link) => (
                            <NavLink
                                key={link.to}
                                to={link.to}
                                className={({ isActive }) =>
                                    `rounded-xl px-3.5 py-1.5 text-xs font-medium font-mono transition-all duration-200 border ${
                                        isActive
                                            ? "border-cyan-400/40 bg-cyan-500/15 text-white shadow-[0_0_15px_rgba(34,211,238,0.2)]"
                                            : "border-transparent text-zinc-400 hover:border-white/10 hover:bg-white/5 hover:text-white"
                                    }`
                                }
                                data-testid={link.testId}
                            >
                                {link.label}
                            </NavLink>
                        ))}
                    </nav>

                    {/* Live Status Pill */}
                    <div className="flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-950/40 px-3.5 py-1.5 shadow-[0_0_15px_rgba(16,185,129,0.15)] backdrop-blur-md">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                        </span>
                        <span className="kicker text-[10px] text-emerald-400 font-semibold tracking-wider">PIPELINE ACTIVE</span>
                    </div>
                </div>
            </header>
            {/* Reserved Spacer to prevent content from jumping under fixed header */}
            <div className="h-16 w-full pointer-events-none" aria-hidden="true" />
        </>
    );
}