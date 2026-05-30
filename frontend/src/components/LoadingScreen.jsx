import { useEffect, useState } from "react";
import { Cpu, Server, Shield, Activity } from "lucide-react";

export default function LoadingScreen({ status }) {
  const [dots, setDots] = useState("");
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    const int = setInterval(() => {
      setDots((d) => (d.length >= 3 ? "" : d + "."));
    }, 400);
    return () => clearInterval(int);
  }, []);

  useEffect(() => {
    if (!status) return;
    const time = new Date().toLocaleTimeString();
    let logMsg = "";

    switch (status.status) {
      case "starting":
        logMsg = `[BOOT] Initializing FastAPI network engine...`;
        break;
      case "profiling":
        logMsg = `[HARDWARE] Scanning target processor architecture... Selected device: ${status.device.toUpperCase()}`;
        break;
      case "loading_trees":
        logMsg = `[METADATA] Discovering semantic tree schemas...`;
        break;
      case "loading_model":
        logMsg = `[NEURAL] Mapping HNSW query vector space (Jina v5 embeddings)...`;
        if (status.cells_loaded) {
          setLogs((prev) => [
            ...prev,
            `[METADATA] Loaded ${status.cells_loaded} cell definitions from trees/`
          ]);
        }
        break;
      case "ready":
        logMsg = `[READY] Neural router initialized. Subsystem online.`;
        break;
      default:
        logMsg = `[STATUS] ${status.message}`;
    }

    if (logMsg && !logs.some(l => l.includes(logMsg))) {
      setLogs((prev) => [...prev, `${time} - ${logMsg}`]);
    }
  }, [status]);

  // Calculate progress percentage
  const getProgress = () => {
    if (!status) return 5;
    switch (status.status) {
      case "starting": return 15;
      case "profiling": return 30;
      case "loading_trees": return 55;
      case "loading_model": return 80;
      case "ready": return 100;
      default: return 10;
    }
  };

  const progress = getProgress();

  return (
    <div className="loading-container">
      {/* Dynamic scan line effect */}
      <div className="scanline" />

      <div className="loading-content glass-panel">
        {/* Futuristic Hexagon Spinner */}
        <div className="spinner-container">
          <svg className="hex-spinner" viewBox="0 0 100 100">
            <polygon
              className="hex-path hex-bg"
              points="50,5 90,25 90,75 50,95 10,75 10,25"
            />
            <polygon
              className="hex-path hex-active"
              points="50,5 90,25 90,75 50,95 10,75 10,25"
              style={{ strokeDashoffset: 340 - (340 * progress) / 100 }}
            />
          </svg>
          <div className="spinner-core">
            <Cpu size={32} className="pulse-icon" />
          </div>
        </div>

        {/* Brand Header */}
        <h1 className="loading-title">NSTL CYBER-LATTICE</h1>
        <p className="loading-subtitle">SYSTEM INITIALIZATION{dots}</p>

        {/* Progress Bar Wrapper */}
        <div className="progress-wrapper">
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="progress-meta">
            <span>Progress: {progress}%</span>
            <span>Device: {status?.device?.toUpperCase() || "CPU"}</span>
          </div>
        </div>

        {/* Current status display */}
        <div className="current-status-box">
          <Activity size={14} className="status-spin" />
          <span className="status-text">{status?.message || "Connecting to core engine..."}</span>
        </div>

        {/* Diagnostic logs */}
        <div className="diagnostic-logs">
          <div className="logs-header">
            <span>DIAGNOSTIC TERMINAL LOGS</span>
            <span className="blink-dot" />
          </div>
          <div className="logs-body">
            {logs.map((log, index) => (
              <div key={index} className="log-line">
                <span className="log-arrow">&gt;</span> {log}
              </div>
            ))}
          </div>
        </div>
      </div>

      <style>{`
        .loading-container {
          position: fixed;
          inset: 0;
          z-index: 99999;
          background: #04060d;
          display: flex;
          align-items: center;
          justify-content: center;
          font-family: "JetBrains Mono", monospace;
          color: #ccd6f6;
          overflow: hidden;
        }

        /* Scanline animation */
        .scanline {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background: linear-gradient(
            to bottom,
            rgba(255,255,255,0),
            rgba(0, 229, 255, 0.03) 50%,
            rgba(255,255,255,0)
          );
          background-size: 100% 4px;
          z-index: 10;
          pointer-events: none;
          animation: scanlineScroll 6s linear infinite;
        }

        .loading-content {
          width: 90%;
          max-width: 500px;
          padding: 40px 30px;
          border: 1px solid rgba(0, 229, 255, 0.15);
          box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7), inset 0 0 20px rgba(0, 229, 255, 0.05);
          text-align: center;
          position: relative;
          z-index: 20;
        }

        .spinner-container {
          position: relative;
          width: 110px;
          height: 110px;
          margin: 0 auto 24px;
        }

        .hex-spinner {
          width: 100%;
          height: 100%;
          transform: rotate(30deg);
        }

        .hex-path {
          fill: none;
          stroke-width: 3;
          stroke-linejoin: round;
        }

        .hex-bg {
          stroke: rgba(0, 229, 255, 0.1);
        }

        .hex-active {
          stroke: #00e5ff;
          stroke-linecap: round;
          stroke-dasharray: 340;
          transition: stroke-dashoffset 0.6s ease-out;
          filter: drop-shadow(0 0 5px rgba(0, 229, 255, 0.7));
        }

        .spinner-core {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          display: flex;
          align-items: center;
          justify-content: center;
          color: #00e5ff;
        }

        .pulse-icon {
          animation: iconPulse 2s infinite ease-in-out;
        }

        .loading-title {
          font-size: 20px;
          font-weight: 700;
          letter-spacing: 4px;
          color: #00e5ff;
          margin-bottom: 8px;
          text-shadow: 0 0 10px rgba(0, 229, 255, 0.3);
        }

        .loading-subtitle {
          font-size: 11px;
          letter-spacing: 2px;
          color: #8a9bbd;
          margin-bottom: 30px;
        }

        .progress-wrapper {
          margin-bottom: 24px;
        }

        .progress-track {
          height: 6px;
          background: rgba(255, 255, 255, 0.05);
          border-radius: 3px;
          overflow: hidden;
          margin-bottom: 8px;
          border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, #ff00aa, #00e5ff);
          border-radius: 3px;
          transition: width 0.4s ease-out;
          box-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
        }

        .progress-meta {
          display: flex;
          justify-content: space-between;
          font-size: 9px;
          color: #8a9bbd;
        }

        .current-status-box {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          background: rgba(0, 229, 255, 0.04);
          border: 1px solid rgba(0, 229, 255, 0.1);
          padding: 10px 16px;
          border-radius: 6px;
          margin-bottom: 24px;
        }

        .status-spin {
          color: #00e5ff;
          animation: spin 3s linear infinite;
        }

        .status-text {
          font-size: 10px;
          letter-spacing: 0.5px;
          color: #ccd6f6;
        }

        .diagnostic-logs {
          background: rgba(0, 0, 0, 0.5);
          border: 1px solid rgba(255, 255, 255, 0.05);
          border-radius: 6px;
          padding: 12px;
          text-align: left;
          height: 110px;
          display: flex;
          flex-direction: column;
        }

        .logs-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 8px;
          color: #8a9bbd;
          border-bottom: 1px solid rgba(255, 255, 255, 0.05);
          padding-bottom: 6px;
          margin-bottom: 8px;
          letter-spacing: 1px;
        }

        .blink-dot {
          width: 4px;
          height: 4px;
          background: #00e5ff;
          border-radius: 50%;
          animation: blink 1.2s infinite;
        }

        .logs-body {
          flex: 1;
          overflow-y: auto;
          font-size: 8.5px;
          color: #8a9bbd;
          scroll-behavior: smooth;
        }

        .log-line {
          margin-bottom: 4px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .log-arrow {
          color: #ff00aa;
        }

        /* Custom Scrollbar for Logs */
        .logs-body::-webkit-scrollbar {
          width: 3px;
        }
        .logs-body::-webkit-scrollbar-thumb {
          background: rgba(0, 229, 255, 0.2);
        }

        /* Animations */
        @keyframes scanlineScroll {
          from { background-position: 0 0; }
          to { background-position: 0 100%; }
        }

        @keyframes iconPulse {
          0%, 100% { transform: scale(1); filter: drop-shadow(0 0 2px rgba(0,229,255,0.2)); }
          50% { transform: scale(1.1); filter: drop-shadow(0 0 8px rgba(0,229,255,0.6)); }
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        @keyframes blink {
          0%, 100% { opacity: 0; }
          50% { opacity: 1; }
        }
      `}</style>
    </div>
  );
}
