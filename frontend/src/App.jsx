import { useState, useEffect } from "react";
import { useStore } from "./store";
import Sidebar from "./components/Sidebar";
import MainArea from "./components/MainArea";
import AppHeader from "./components/AppHeader";
import StatusBar from "./components/StatusBar";
import NodeTooltip from "./components/NodeTooltip";
import LoadingScreen from "./components/LoadingScreen";
import DevMenu from "./components/DevMenu";
import { fetchHealth, fetchCells, fetchStatus, initializeEngine, fetchAvailableModels } from "./hooks/useApi";
import { Terminal, Map, Code2 } from "lucide-react";

export default function App() {
  const [loading, setLoading] = useState(true);
  const [bootStatus, setBootStatus] = useState({
    status: "connecting",
    message: "Initiating handshake with engine...",
    device: "cpu",
    cells_loaded: 0,
  });

  const [selectedProfile, setSelectedProfile] = useState("C"); // Default to C (Dedicated Embedder + LLM)
  // B-2 fix: default to empty string — the first /api/models call fills these
  // with whatever is actually installed instead of hardcoded names.
  const [embedderModel, setEmbedderModel] = useState("");
  const [llmModel, setLlmModel] = useState("");
  const [embedderDevice, setEmbedderDevice] = useState("auto");
  const [llmDevice, setLlmDevice] = useState("auto");
  const [treesStorage] = useState("ram");
  const [availableModels, setAvailableModels] = useState({ embedders: [], llms: [] });

  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);
  const [mobileView, setMobileView] = useState("chat"); // 'chat' | 'main'

  const setCells = useStore((s) => s.setCells);
  const setApiStatus = useStore((s) => s.setApiStatus);
  const logSystemEvent = useStore((s) => s.logSystemEvent);
  const setHardwareDevice = useStore((s) => s.setHardwareDevice);

  // Boot sequence
  useEffect(() => {
    logSystemEvent("Application initialized, polling engine status...", "UI");

    // B-2 + G-1 fix: helper that always keeps availableModels up to date
    const refreshModels = async () => {
      try {
        const models = await fetchAvailableModels();
        setAvailableModels(models);
        // B-2 fix: if model names are still empty (first boot), pick the first available
        setEmbedderModel((cur) => (cur || (models.embedders[0] ?? "")));
        setLlmModel((cur) => (cur || (models.llms[0] ?? "")));
      } catch (err) {
        console.error("Failed to fetch models", err);
      }
    };

    const monitorStatus = async () => {
      try {
        const res = await fetchStatus();
        if (res.ok) {
          const data = await res.json();
          setBootStatus(data);
          if (data.device) setHardwareDevice(data.device);
          logSystemEvent(`Engine status: ${data.status}`, "API");

          if (data.status === "ready") {
            clearInterval(pollInterval);
            // G-1 fix: always refresh models on ready so pickers are never empty
            await refreshModels();
            setTimeout(() => {
              setLoading(false);
              logSystemEvent("Dashboard live.", "STATE");
            }, 500);
          } else if (data.status === "uninitialized") {
            clearInterval(pollInterval);
            await refreshModels();
            logSystemEvent("Auto-initializing engine...", "API");
            setBootStatus((p) => ({ ...p, message: "Bootstrapping neural models...", status: "loading" }));
            // B-1 fix: use selectedProfile state variable, NOT a hardcoded "B"
            await initializeEngine(selectedProfile, embedderModel, llmModel, "auto", "auto", "ram");
            logSystemEvent("Initialization request sent, waiting for backend...", "API");
            // Resume polling to detect when ready
            pollInterval = setInterval(monitorStatus, 850);
          } else if (data.status === "loading" || data.status === "initializing") {
            // G-1 fix: refresh models if available list is still empty (e.g. page reloaded mid-init)
            setAvailableModels((prev) => {
              if (prev.embedders.length === 0 && prev.llms.length === 0) {
                refreshModels();
              }
              return prev;
            });
          } else if (data.status === "error") {
            // Hard stop — do not retry automatically to avoid infinite loop
            clearInterval(pollInterval);
            logSystemEvent(`Engine initialization failed: ${data.message}`, "ERROR");
            setBootStatus((p) => ({
              ...p,
              status: "error",
              message: `Init failed: ${data.message || "unknown error"}. Click Retry to try again.`,
            }));
          }
          // other states: keep polling
        }
      } catch (err) {
        console.debug("Engine not reachable yet:", err.message);
        setBootStatus({ status: "connecting", message: "Waiting for backend...", device: "cpu", cells_loaded: 0 });
      }
    };

    // B-3 fix: assign pollInterval BEFORE calling monitorStatus() so the
    // interval reference exists when the very first async callback resolves.
    // Previously, the first call could hit "uninitialized", set a new interval
    // from inside the callback, and then the line below would set ANOTHER one —
    // creating two concurrent polling loops and duplicate /api/initialize calls.
    let pollInterval = setInterval(monitorStatus, 850);
    monitorStatus(); // fire immediately too
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
        } catch (err) {
          console.warn("Post-boot health check failed:", err.message);
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

  const handleProfileSwap = async (newProfile) => {
    logSystemEvent(`Hot-swapping profile to: ${newProfile}`, "API");
    setSelectedProfile(newProfile);
    try {
      await initializeEngine(newProfile, embedderModel, llmModel, embedderDevice, llmDevice, treesStorage);
      const cells = await fetchCells();
      setCells(cells);
      logSystemEvent("Profile swap complete.", "API");
    } catch (err) {
      logSystemEvent(`Profile swap failed: ${err.message}`, "API");
    }
  };


  if (loading) return <LoadingScreen status={bootStatus} />;

  return (
    <div className="app-shell">
      <AppHeader
        embedderModel={embedderModel}
        llmModel={llmModel}
        onModelSwap={handleModelSwap}
        availableModels={availableModels}
        selectedProfile={selectedProfile}
        onProfileChange={handleProfileSwap}
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

      <DevMenu
        embedderDevice={embedderDevice}
        setEmbedderDevice={setEmbedderDevice}
        llmDevice={llmDevice}
        setLlmDevice={setLlmDevice}
        onHardwareApply={async () => {
          logSystemEvent(`Applying hardware settings: EMB=${embedderDevice}, LLM=${llmDevice}`, "API");
          try {
            await initializeEngine(selectedProfile, embedderModel, llmModel, embedderDevice, llmDevice, treesStorage);
            const cells = await fetchCells();
            setCells(cells);
            logSystemEvent("Hardware settings applied successfully.", "API");
          } catch (err) {
            logSystemEvent(`Hardware apply failed: ${err.message}`, "API");
          }
        }}
      />
    </div>
  );
}
