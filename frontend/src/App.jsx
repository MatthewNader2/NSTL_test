import { useState, useEffect } from "react";
import { useStore } from "./store";
import Sidebar from "./components/Sidebar";
import MainArea from "./components/MainArea";
import AppHeader from "./components/AppHeader";
import StatusBar from "./components/StatusBar";
import NodeTooltip from "./components/NodeTooltip";
import LoadingScreen from "./components/LoadingScreen";
import DevMenu from "./components/DevMenu";
import { fetchHealth, fetchCells, fetchStatus, initializeEngine } from "./hooks/useApi";
import { Terminal, Map, Code2 } from "lucide-react";

export default function App() {
  const [loading, setLoading] = useState(true);
  const [bootStatus, setBootStatus] = useState({
    status: "connecting",
    message: "Initiating handshake with engine...",
    device: "cpu",
    cells_loaded: 0,
  });

  const [selectedProfile] = useState("A");
  const [embedderModel, setEmbedderModel] = useState("jina-embeddings-v5-text-nano");
  const [llmModel, setLlmModel] = useState("qwen2.5-coder-1.5b-instruct");
  const [embedderDevice] = useState("auto");
  const [llmDevice] = useState("auto");
  const [treesStorage] = useState("ram");

  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);
  const [mobileView, setMobileView] = useState("chat"); // 'chat' | 'main'

  const setCells = useStore((s) => s.setCells);
  const setApiStatus = useStore((s) => s.setApiStatus);
  const logSystemEvent = useStore((s) => s.logSystemEvent);
  const setHardwareDevice = useStore((s) => s.setHardwareDevice);

  // Boot sequence
  useEffect(() => {
    let pollInterval;
    logSystemEvent("Application initialized, polling engine status...", "UI");

    const monitorStatus = async () => {
      try {
        const res = await fetchStatus();
        if (res.ok) {
          const data = await res.json();
          setBootStatus(data);
          if (data.device) setHardwareDevice(data.device);
          logSystemEvent(`Engine status: ${data.status}`, "API");

          if (data.status === "ready" || data.status === "uninitialized") {
            clearInterval(pollInterval);
            if (data.status === "uninitialized") {
              logSystemEvent("Auto-initializing engine...", "API");
              setBootStatus((p) => ({ ...p, message: "Bootstrapping neural models...", status: "connecting" }));
              await initializeEngine("A", embedderModel, llmModel, "auto", "auto", "ram");
              logSystemEvent("Auto-initialization complete.", "API");
            }
            setTimeout(() => {
              setLoading(false);
              logSystemEvent("Dashboard live.", "STATE");
            }, 500);
          }
        }
      } catch {
        setBootStatus({ status: "connecting", message: "Waiting for backend...", device: "cpu", cells_loaded: 0 });
      }
    };

    monitorStatus();
    pollInterval = setInterval(monitorStatus, 850);
    return () => clearInterval(pollInterval);
  }, []);

  // Post-boot: fetch cells and setup resize
  useEffect(() => {
    if (!loading) {
      (async () => {
        try {
          const res = await fetchHealth();
          if (res.ok) {
            setApiStatus("live");
            const cells = await fetchCells();
            setCells(cells);
            logSystemEvent(`Fetched ${cells.length} cells`, "API");
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
    };
    window.addEventListener("resize", handleResize);
    handleResize();
    return () => window.removeEventListener("resize", handleResize);
  }, [loading]);

  const handleModelSwap = async (newEmbedder, newLlm) => {
    logSystemEvent(`Hot-swapping: ${newEmbedder} / ${newLlm}`, "API");
    setEmbedderModel(newEmbedder);
    setLlmModel(newLlm);
    try {
      await initializeEngine(selectedProfile, newEmbedder, newLlm, embedderDevice, llmDevice, treesStorage);
      const cells = await fetchCells();
      setCells(cells);
      logSystemEvent("Hot-swap complete.", "API");
    } catch (err) {
      logSystemEvent(`Hot-swap failed: ${err.message}`, "API");
    }
  };

  if (loading) return <LoadingScreen status={bootStatus} />;

  return (
    <div className="app-shell">
      <AppHeader
        embedderModel={embedderModel}
        llmModel={llmModel}
        onModelSwap={handleModelSwap}
      />

      <div className="app-body">
        {/* Sidebar */}
        <Sidebar
          className={isMobile ? (mobileView === "chat" ? "mobile-active" : "") : ""}
        />

        {/* Main Area (tabs: code + 3D) */}
        <MainArea
          className={isMobile ? (mobileView === "main" ? "mobile-active" : "") : ""}
        />
      </div>

      <StatusBar />
      {!isMobile && <NodeTooltip />}

      {/* Mobile bottom nav */}
      {isMobile && (
        <nav className="mobile-nav">
          <button
            className={`mobile-nav-btn ${mobileView === "chat" ? "active" : ""}`}
            onClick={() => setMobileView("chat")}
          >
            <Terminal size={20} />
            <span>Chat</span>
          </button>
          <button
            className={`mobile-nav-btn ${mobileView === "main" ? "active" : ""}`}
            onClick={() => setMobileView("main")}
          >
            <Map size={20} />
            <span>Workspace</span>
          </button>
        </nav>
      )}

      <DevMenu />
    </div>
  );
}
