import { useState, useCallback, useRef, useEffect } from "react";
import { useStore } from "../store";
import { runPrompt } from "../hooks/useApi";
import {
  Send,
  X,
  ChevronDown,
  ChevronUp,
  Cpu,
  RefreshCw,
  Terminal,
  User,
  Sparkles,
  HelpCircle,
} from "lucide-react";

export default function LeftPanel({ open, isMobile, mobileActive, onToggle }) {
  const [prompt, setPrompt] = useState("");
  const [localThinkingId, setLocalThinkingId] = useState(null);
  
  // Store values & actions
  const chatHistory = useStore((s) => s.chatHistory);
  const addHistoryItem = useStore((s) => s.addHistoryItem);
  const updateHistoryItem = useStore((s) => s.updateHistoryItem);
  const activeHistoryId = useStore((s) => s.activeHistoryId);
  const setActiveHistoryId = useStore((s) => s.setActiveHistoryId);
  
  const setActivePath = useStore((s) => s.setActivePath);
  const setVirtualEdges = useStore((s) => s.setVirtualEdges);
  const setGeneratedCode = useStore((s) => s.setGeneratedCode);
  const setLogs = useStore((s) => s.setLogs);
  const logSystemEvent = useStore((s) => s.logSystemEvent);

  const historyEndRef = useRef(null);

  // Auto scroll to bottom of chat history when items change
  useEffect(() => {
    historyEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, localThinkingId]);

  // Collapsible accordion state for thinking processes inside chat list
  const [expandedAccordions, setExpandedAccordions] = useState({});

  const toggleAccordion = (id, e) => {
    e.stopPropagation(); // Prevent selecting the message bubble
    setExpandedAccordions((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
    logSystemEvent(`Toggled thinking accordion for item ${id}`, "UI");
  };

  const handleRun = useCallback(async (customPrompt) => {
    const query = customPrompt || prompt;
    if (!query.trim()) return;

    const id = Date.now();
    logSystemEvent(`Initiating pipeline execution for prompt: "${query}"`, "API");

    // 1. Add interactive thinking item in chat history
    const tempItem = {
      id,
      prompt: query,
      isThinking: true,
      logs: [{ type: "system", msg: "Initializing engine handshake...", time: new Date().toLocaleTimeString() }],
      path: [],
      code: "",
      virtualEdges: new Set(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    addHistoryItem(tempItem);
    setActiveHistoryId(id);
    setLocalThinkingId(id);
    setPrompt("");

    try {
      const data = await runPrompt(query);
      
      // Update global states
      setActivePath(data.path);
      setVirtualEdges(new Set(data.virtual_edges));
      setGeneratedCode(data.code);
      setLogs(data.logs);

      // 2. Resolve thinking process and cache results in history
      updateHistoryItem(id, {
        isThinking: false,
        logs: data.logs,
        path: data.path,
        code: data.code,
        virtualEdges: new Set(data.virtual_edges),
      });

      logSystemEvent(`Pipeline resolution complete. Unified ${data.path.length} execution cells.`, "ENGINE");
    } catch (err) {
      const errorLogs = [{ type: "warn", msg: `[API ERROR] ${err.message}`, time: new Date().toLocaleTimeString() }];
      setLogs(errorLogs);
      
      updateHistoryItem(id, {
        isThinking: false,
        logs: errorLogs,
        path: [],
        code: `# Error: ${err.message}`,
        virtualEdges: new Set(),
      });

      logSystemEvent(`Pipeline execution failed: ${err.message}`, "API");
    } finally {
      setLocalThinkingId(null);
    }
  }, [prompt, addHistoryItem, updateHistoryItem, setActiveHistoryId, setActivePath, setVirtualEdges, setGeneratedCode, setLogs, logSystemEvent]);

  // Click handler to restore a past prompt run
  const handleSelectHistoryItem = (item) => {
    if (item.isThinking) return;
    
    setActiveHistoryId(item.id);
    setActivePath(item.path);
    setVirtualEdges(item.virtualEdges);
    setGeneratedCode(item.code);
    setLogs(item.logs);
    
    logSystemEvent(`Restored past query results for ID: ${item.id}`, "STATE");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleRun();
    }
  };

  // Sophisticated tech/data-science suggestions
  const suggestions = [
    "Ingest a Parquet file, clean missing data, and train a RandomForest classifier",
    "Create a Flask web service that queries a database and returns JSON results",
    "Load a CSV file, group by user score, and plot the distribution using matplotlib",
    "Run a complete text preprocessing pipeline, build embeddings, and index with FAISS",
  ];

  return (
    <div
      className={`left-panel glass-panel ${!isMobile && !open ? "collapsed" : ""} ${isMobile && mobileActive ? "mobile-active" : ""}`}
      style={{
        borderRight: open ? "1px solid var(--glass-border)" : "none",
      }}
    >
      {open && (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            height: "100%",
            padding: "0.75rem",
            minHeight: 0,
            minWidth: 0,
          }}
        >
          {/* Panel Header */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "0.75rem",
              borderBottom: "1px solid var(--glass-border)",
              paddingBottom: "0.5rem",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "0.375rem" }}>
              <Cpu size={14} color="var(--accent)" />
              <span
                style={{
                  fontSize: "0.7rem",
                  fontWeight: "bold",
                  color: "var(--text-secondary)",
                  letterSpacing: 1.5,
                }}
              >
                LATTICE CONSOLE
              </span>
            </div>
            <button
              onClick={onToggle}
              style={{ color: "var(--text-secondary)", display: "flex", padding: "2px" }}
              title="Collapse Panel"
            >
              <X size={14} />
            </button>
          </div>

          {/* Interactive Chat Stream & Collapsible Logs */}
          <div
            style={{
              flex: 1,
              overflowY: "auto",
              marginBottom: "0.75rem",
              display: "flex",
              flexDirection: "column",
              gap: "1rem",
              paddingRight: "4px",
              minHeight: 0,
            }}
          >
            {chatHistory.length === 0 ? (
              <div
                style={{
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  textAlign: "center",
                  color: "var(--text-secondary)",
                  padding: "1rem",
                  fontSize: "0.8rem",
                  gap: "0.5rem",
                }}
              >
                <Sparkles size={24} color="var(--accent)" style={{ opacity: 0.6 }} />
                <span>No active sessions. Select a suggestion below or describe a data pipeline to compile.</span>
              </div>
            ) : (
              chatHistory.map((item) => {
                const isSelected = activeHistoryId === item.id;
                const isThinking = item.isThinking;
                const showThinking = expandedAccordions[item.id] !== false; // Default to true

                return (
                  <div
                    key={item.id}
                    onClick={() => handleSelectHistoryItem(item)}
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: "0.5rem",
                      cursor: isThinking ? "default" : "pointer",
                      opacity: isThinking ? 0.85 : 1,
                    }}
                  >
                    {/* User Prompt Speech Bubble */}
                    <div style={{ display: "flex", justifyContent: "flex-end" }}>
                      <div
                        style={{
                          background: isSelected ? "rgba(0, 229, 255, 0.15)" : "rgba(255, 255, 255, 0.04)",
                          border: isSelected ? "1px solid rgba(0, 229, 255, 0.35)" : "1px solid rgba(255, 255, 255, 0.08)",
                          padding: "8px 12px",
                          borderRadius: "12px 12px 0 12px",
                          maxWidth: "85%",
                          fontSize: "0.75rem",
                          color: "var(--text-primary)",
                          boxShadow: isSelected ? "0 0 10px rgba(0, 229, 255, 0.08)" : "none",
                          transition: "all 0.2s ease",
                        }}
                      >
                        <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "4px", fontSize: "0.6rem", opacity: 0.6 }}>
                          <User size={10} />
                          <span>USER • {item.timestamp}</span>
                        </div>
                        <div style={{ wordBreak: "break-word" }}>{item.prompt}</div>
                      </div>
                    </div>

                    {/* Assistant Response & Collapsible Thinking Logs */}
                    <div style={{ display: "flex", justifyContent: "flex-start" }}>
                      <div
                        style={{
                          background: "rgba(8, 14, 28, 0.6)",
                          border: "1px solid var(--glass-border)",
                          padding: "10px",
                          borderRadius: "12px 12px 12px 0",
                          width: "95%",
                          fontSize: "0.75rem",
                          color: "var(--text-primary)",
                        }}
                      >
                        {/* Thinking Header (Accordion Trigger) */}
                        <div
                          onClick={(e) => toggleAccordion(item.id, e)}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            cursor: "pointer",
                            paddingBottom: showThinking ? "6px" : "0",
                            borderBottom: showThinking ? "1px solid rgba(255, 255, 255, 0.05)" : "none",
                            color: isThinking ? "var(--accent)" : "var(--text-secondary)",
                          }}
                        >
                          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                            {isThinking ? (
                              <RefreshCw size={12} className="spin-animation" style={{ animation: "spin 1.5s linear infinite" }} />
                            ) : (
                              <Terminal size={12} />
                            )}
                            <span style={{ fontWeight: "600", fontSize: "0.7rem", letterSpacing: 0.5 }}>
                              {isThinking ? "Thinking process..." : "View Thinking Logs"}
                            </span>
                          </div>
                          {showThinking ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                        </div>

                        {/* Collapsed/Expanded Log Content */}
                        {showThinking && (
                          <div
                            style={{
                              marginTop: "6px",
                              maxHeight: "160px",
                              overflowY: "auto",
                              fontSize: "0.65rem",
                              color: "var(--text-secondary)",
                              display: "flex",
                              flexDirection: "column",
                              gap: "4px",
                              background: "rgba(0,0,0,0.3)",
                              padding: "6px",
                              borderRadius: "6px",
                              fontFamily: "var(--font-mono)",
                            }}
                          >
                            {item.logs?.length === 0 ? (
                              <div style={{ opacity: 0.5, fontStyle: "italic" }}>Handshaking...</div>
                            ) : (
                              item.logs.map((log, li) => {
                                let logColor = "#8a9bbd";
                                if (log.type === "system") logColor = "#c678dd";
                                else if (log.type === "warn" || log.type === "error") logColor = "#e06c75";
                                else if (log.type === "info") logColor = "#61afef";
                                
                                return (
                                  <div key={li} style={{ color: logColor, wordBreak: "break-word" }}>
                                    <span style={{ opacity: 0.4 }}>[{log.time || ""}]</span> {log.msg}
                                  </div>
                                );
                              })
                            )}
                          </div>
                        )}

                        {/* Completed Status Summary */}
                        {!isThinking && item.path?.length > 0 && (
                          <div
                            style={{
                              marginTop: "8px",
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "space-between",
                              fontSize: "0.7rem",
                              color: "var(--accent)",
                            }}
                          >
                            <span>⬡ Lattice Path: {item.path.length} cells compiled</span>
                            <span style={{ opacity: 0.6, fontSize: "0.6rem" }}>
                              {item.code ? "CODE COMPILED" : "EMPTY"}
                            </span>
                          </div>
                        )}
                        
                        {!isThinking && item.path?.length === 0 && (
                          <div
                            style={{
                              marginTop: "8px",
                              fontSize: "0.7rem",
                              color: "#e06c75",
                            }}
                          >
                            ⚠️ Unification failed. No valid execution paths found.
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })
            )}
            {/* Quick Sophisticated Suggestions */}
            {chatHistory.length === 0 && (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "4px",
                  marginTop: "auto",
                  padding: "0 4px",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "4px", opacity: 0.6 }}>
                  <HelpCircle size={10} color="var(--text-secondary)" />
                  <span style={{ fontSize: "0.6rem", color: "var(--text-secondary)", fontWeight: "bold" }}>
                    SUGGESTIONS
                  </span>
                </div>
                <div
                  style={{
                    display: "flex",
                    gap: "0.375rem",
                    flexWrap: "wrap",
                  }}
                >
                  {suggestions.map((ex) => (
                    <button
                      key={ex}
                      onClick={() => {
                        setPrompt(ex);
                        logSystemEvent(`Clicked prompt suggestion: "${ex.substring(0, 30)}..."`, "UI");
                      }}
                      style={{
                        fontSize: "0.6rem",
                        padding: "4px 8px",
                        background: "rgba(0, 229, 255, 0.06)",
                        border: "1px solid rgba(0, 229, 255, 0.12)",
                        borderRadius: 6,
                        color: "var(--text-secondary)",
                        textAlign: "left",
                        width: "100%",
                        display: "block",
                        whiteSpace: "normal",
                        lineHeight: "1.3",
                      }}
                    >
                      {ex}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div ref={historyEndRef} />
          </div>

          {/* Prompt Input TextArea */}
          <div style={{ display: "flex", gap: "0.375rem", alignItems: "flex-end" }}>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe your pipeline intent..."
              rows={2}
              style={{
                flex: 1,
                resize: "none",
                background: "rgba(0,0,0,0.4)",
                border: "1px solid var(--glass-border)",
                borderRadius: 6,
                padding: "8px",
                fontSize: "0.75rem",
                color: "var(--text-primary)",
                outline: "none",
                lineHeight: "1.4",
              }}
            />
            <button
              onClick={() => handleRun()}
              disabled={!prompt.trim() || localThinkingId !== null}
              style={{
                padding: "8px 12px",
                background: "var(--accent)",
                borderRadius: 6,
                color: "#000",
                fontWeight: 700,
                opacity: prompt.trim() && localThinkingId === null ? 1 : 0.4,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                height: "36px",
              }}
            >
              <Send size={14} />
            </button>
          </div>
        </div>
      )}
      
      {/* Toggle button when collapsed */}
      {!open && (
        <button
          onClick={onToggle}
          style={{
            position: "absolute",
            top: "0.5rem",
            left: "0.5rem",
            zIndex: 10,
            color: "var(--text-secondary)",
          }}
        >
          <Sparkles size={16} />
        </button>
      )}
    </div>
  );
}
