import { useState, useEffect } from "react";
import { useStore } from "./store";
import TitleBar from "./components/TitleBar";
import LeftPanel from "./components/LeftPanel";
import RightPanel from "./components/RightPanel";
import StatusBar from "./components/StatusBar";
import ThreeScene from "./components/ThreeScene";
import NodeTooltip from "./components/NodeTooltip";
import LoadingScreen from "./components/LoadingScreen";
import { fetchHealth, fetchCells, fetchStatus, getApiBase, setApiBase } from "./hooks/useApi";
import { Terminal, Map, Code } from "lucide-react";

export default function App() {
  // Loading & Boot states
  const [loading, setLoading] = useState(true);
  const [bootStatus, setBootStatus] = useState({
    status: "connecting",
    message: "Initiating handshake with engine...",
    device: "cpu",
    cells_loaded: 0
  });

  // Settings states
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [apiInputUrl, setApiInputUrl] = useState(getApiBase());

  // Panel Open States (Desktop)
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);

  // Mobile navigation view state ('console', 'map', 'code')
  const [mobileView, setMobileView] = useState("map");
  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);

  const setCells = useStore((s) => s.setCells);
  const setApiStatus = useStore((s) => s.setApiStatus);

  // 📡 Real-time background initialization monitor
  useEffect(() => {
    let pollInterval;
    
    const monitorStatus = async () => {
      try {
        const res = await fetchStatus();
        if (res.ok) {
          const data = await res.json();
          setBootStatus(data);
          if (data.status === "ready") {
            clearInterval(pollInterval);
            // Smoothly remove loading screen
            setTimeout(() => {
              setLoading(false);
            }, 600);
          }
        }
      } catch (err) {
        setBootStatus({
          status: "connecting",
          message: "Waiting for server to respond...",
          device: "cpu",
          cells_loaded: 0
        });
      }
    };

    monitorStatus();
    pollInterval = setInterval(monitorStatus, 800);
    return () => clearInterval(pollInterval);
  }, []);

  // 🔍 Fetch cell metadata and setup scaling listener
  useEffect(() => {
    if (!loading) {
      (async () => {
        try {
          const res = await fetchHealth();
          if (res.ok) {
            setApiStatus("live");
            const data = await fetchCells();
            setCells(data.cells || []);
          } else {
            setApiStatus("offline");
          }
        } catch {
          setApiStatus("offline");
        }
      })();
    }

    const handleResize = () => {
      const mobile = window.innerWidth <= 768;
      setIsMobile(mobile);
      
      if (mobile) {
        setLeftOpen(true);
        setRightOpen(true);
        // Mobile scaling: base CSS width of 375px maps to 15px font
        const baseMobileWidth = 375;
        const scale = Math.max(0.85, Math.min(1.2, window.innerWidth / baseMobileWidth));
        document.documentElement.style.fontSize = `${scale * 15}px`;
      } else {
        // Desktop scaling: scales with physical width, canceling system dpr
        const dpr = window.devicePixelRatio || 1;
        const physicalWidth = window.innerWidth * dpr;
        const rawScale = physicalWidth / 1920;
        const scale = Math.max(0.75, Math.min(2.0, rawScale)); // Clamp scale factor
        const cssFontSize = (scale * 16) / dpr;
        document.documentElement.style.fontSize = `${cssFontSize}px`;
      }
    };

    window.addEventListener("resize", handleResize);
    handleResize(); // Execute once on mount

    return () => window.removeEventListener("resize", handleResize);
  }, [loading]);

  if (loading) {
    return <LoadingScreen status={bootStatus} />;
  }

  return (
    <div className="app-container">
      <TitleBar onSettingsClick={() => {
        setApiInputUrl(getApiBase());
        setSettingsOpen(true);
      }} />
      <div className="main-layout">
        {/* Left Panel (Console) */}
        <div
          className={`left-panel glass-panel ${!isMobile && !leftOpen ? "collapsed" : ""} ${isMobile && mobileView === "console" ? "mobile-active" : ""}`}
        >
          <LeftPanel
            open={isMobile ? true : leftOpen}
            onToggle={() => setLeftOpen(!leftOpen)}
          />
        </div>

        {/* Center 3D Canvas */}
        <div
          className="scene-wrapper"
          style={{
            display: isMobile && mobileView !== "map" ? "none" : "block",
          }}
        >
          <ThreeScene />
        </div>

        {/* Right Panel (Inspector/Code) */}
        <div
          className={`right-panel glass-panel ${!isMobile && !rightOpen ? "collapsed" : ""} ${isMobile && mobileView === "code" ? "mobile-active" : ""}`}
        >
          <RightPanel
            open={isMobile ? true : rightOpen}
            onToggle={() => setRightOpen(!rightOpen)}
          />
        </div>

        {/* 📱 MOBILE BOTTOM NAVIGATION */}
        <div className="bottom-nav">
          <button
            className={`nav-btn ${mobileView === "console" ? "active" : ""}`}
            onClick={() => setMobileView("console")}
          >
            <Terminal size={20} />
            <span>Console</span>
          </button>

          <button
            className={`nav-btn ${mobileView === "map" ? "active" : ""}`}
            onClick={() => setMobileView("map")}
          >
            <Map size={20} />
            <span>Lattice</span>
          </button>

          <button
            className={`nav-btn ${mobileView === "code" ? "active" : ""}`}
            onClick={() => setMobileView("code")}
          >
            <Code size={20} />
            <span>Output</span>
          </button>
        </div>
      </div>
      <StatusBar />
      {!isMobile && <NodeTooltip />}

      {/* 📡 Connection Config Modal */}
      {settingsOpen && (
        <div style={{
          position: "fixed",
          inset: 0,
          background: "rgba(4, 6, 12, 0.85)",
          backdropFilter: "blur(8px)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 100000,
          fontFamily: "var(--font-mono)"
        }}>
          <div className="glass-panel" style={{
            width: "90%",
            maxWidth: "400px",
            padding: "20px",
            border: "1px solid var(--accent)",
            boxShadow: "0 10px 40px rgba(0,0,0,0.8)"
          }}>
            <h3 style={{ color: "var(--accent)", marginBottom: "16px", fontSize: "1rem" }}>⬡ Connection Settings</h3>
            <div style={{ marginBottom: "16px" }}>
              <label style={{ display: "block", fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: "6px" }}>
                FastAPI Server IP Address
              </label>
              <input
                type="text"
                value={apiInputUrl}
                onChange={(e) => setApiInputUrl(e.target.value)}
                placeholder="e.g. http://192.168.1.15:8000"
                style={{
                  width: "100%",
                  padding: "10px",
                  background: "rgba(0,0,0,0.4)",
                  border: "1px solid var(--glass-border)",
                  borderRadius: "6px",
                  color: "#fff",
                  fontSize: "0.85rem",
                  outline: "none"
                }}
              />
            </div>
            <div style={{ display: "flex", gap: "10px", justifyContent: "flex-end" }}>
              <button
                onClick={() => setSettingsOpen(false)}
                style={{
                  padding: "8px 14px",
                  background: "rgba(255,255,255,0.08)",
                  borderRadius: "6px",
                  fontSize: "0.8rem",
                  color: "var(--text-secondary)"
                }}
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setApiBase(apiInputUrl);
                  setSettingsOpen(false);
                  window.location.reload();
                }}
                style={{
                  padding: "8px 14px",
                  background: "var(--accent)",
                  borderRadius: "6px",
                  fontSize: "0.8rem",
                  color: "#000",
                  fontWeight: "bold"
                }}
              >
                Save & Connect
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
