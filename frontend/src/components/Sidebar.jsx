import { useState, useCallback, useRef, useEffect } from "react";
import { useStore } from "../store";
import { runPrompt, fetchCells } from "../hooks/useApi";
import {
  Send, X, ChevronDown, ChevronUp, RefreshCw, Terminal,
  User, Sparkles, HelpCircle, Plus, MessageSquareReply, Bot,
} from "lucide-react";

const SUGGESTIONS = [
  "Load a CSV, remove nulls, normalize numeric columns, train a logistic regression and print accuracy.",
  "Build a Flask REST API with SQLite — /users GET + POST endpoints.",
  "Read a CSV, group by category, plot mean values per group as a bar chart.",
  "Embed sentences with a transformer, index into FAISS for similarity search.",
];

export default function Sidebar({ className = "" }) {
  const [prompt, setPrompt] = useState("");
  const [localThinkingId, setLocalThinkingId] = useState(null);
  const [replyContext, setReplyContext] = useState(null);
  const [expandedAccordions, setExpandedAccordions] = useState({});
  const [showAll, setShowAll] = useState(false);

  const chatHistory = useStore((s) => s.chatHistory);
  const addHistoryItem = useStore((s) => s.addHistoryItem);
  const updateHistoryItem = useStore((s) => s.updateHistoryItem);
  const activeHistoryId = useStore((s) => s.activeHistoryId);
  const setActiveHistoryId = useStore((s) => s.setActiveHistoryId);
  const clearHistory = useStore((s) => s.clearHistory);
  const setCells = useStore((s) => s.setCells);
  const setActivePath = useStore((s) => s.setActivePath);
  const setVirtualEdges = useStore((s) => s.setVirtualEdges);
  const setGeneratedCode = useStore((s) => s.setGeneratedCode);
  const setLogs = useStore((s) => s.setLogs);
  const logSystemEvent = useStore((s) => s.logSystemEvent);

  const historyEndRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    historyEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, localThinkingId]);

  const toggleAccordion = (id, e) => {
    e.stopPropagation();
    setExpandedAccordions((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleRun = useCallback(async (customPrompt) => {
    const query = customPrompt || prompt;
    if (!query.trim()) return;

    const id = Date.now();
    logSystemEvent(`Running: "${query.substring(0, 60)}"`, "API");

    const tempItem = {
      id, prompt: query, isThinking: true,
      logs: [{ type: "system", msg: "Handshaking...", time: new Date().toLocaleTimeString() }],
      path: [], code: "", virtualEdges: new Set(),
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    addHistoryItem(tempItem);
    setActiveHistoryId(id);
    setLocalThinkingId(id);
    setPrompt("");

    let payload = query;
    if (replyContext) {
      payload = `Previous Task: ${replyContext.prompt}\n\nCurrent Implementation:\n${replyContext.code}\n\nFollow-up: ${query}`;
    }

    try {
      const data = await runPrompt(payload);
      setReplyContext(null);
      setActivePath(data.path);
      setVirtualEdges(new Set(data.virtual_edges));
      setGeneratedCode(data.code);
      setLogs(data.logs);
      const cells = await fetchCells();
      setCells(cells);
      updateHistoryItem(id, {
        isThinking: false,
        logs: data.logs,
        path: data.path,
        code: data.code,
        virtualEdges: new Set(data.virtual_edges),
      });
      logSystemEvent(`Done. ${data.path.length} cells unified.`, "ENGINE");
    } catch (err) {
      const errorLogs = [{ type: "warn", msg: `[API ERROR] ${err.message}`, time: new Date().toLocaleTimeString() }];
      setLogs(errorLogs);
      updateHistoryItem(id, { isThinking: false, logs: errorLogs, path: [], code: `# Error: ${err.message}`, virtualEdges: new Set() });
      logSystemEvent(`Failed: ${err.message}`, "API");
    } finally {
      setLocalThinkingId(null);
    }
  }, [prompt, replyContext, addHistoryItem, updateHistoryItem, setActiveHistoryId, setActivePath, setVirtualEdges, setGeneratedCode, setLogs, logSystemEvent]);

  const handleSelectHistory = (item) => {
    if (item.isThinking) return;
    setActiveHistoryId(item.id);
    setActivePath(item.path);
    setVirtualEdges(item.virtualEdges);
    setGeneratedCode(item.code);
    setLogs(item.logs);
    logSystemEvent(`Restored session ${item.id}`, "STATE");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleRun();
    }
  };

  // Auto-grow textarea
  const handleInput = (e) => {
    setPrompt(e.target.value);
    const el = textareaRef.current;
    if (el) { el.style.height = "auto"; el.style.height = Math.min(el.scrollHeight, 140) + "px"; }
  };

  return (
    <aside className={`sidebar ${className}`}>
      {/* Header */}
      <div className="sidebar-header">
        <div className="sidebar-title">
          <Sparkles size={14} color="var(--cyan)" />
          <span>LATTICE CONSOLE</span>
        </div>
        <div className="sidebar-actions">
          <button
            className="icon-btn"
            title="New Chat"
            onClick={() => { clearHistory(); setReplyContext(null); logSystemEvent("New chat started", "UI"); }}
          >
            <Plus size={15} />
          </button>
        </div>
      </div>

      {/* Chat Scroll */}
      <div className="chat-scroll">
        {chatHistory.length === 0 ? (
          <div className="chat-empty">
            <div className="chat-empty-icon"><Sparkles size={22} /></div>
            <div className="chat-empty-title">Start a session</div>
            <div className="chat-empty-sub">Describe a data pipeline and the neural lattice will compile it into executable code.</div>
          </div>
        ) : (
          <>
            {chatHistory.length > 200 && !showAll && (
              <button className="show-earlier-btn" onClick={() => setShowAll(true)}>
                Show {chatHistory.length - 200} earlier messages
              </button>
            )}
            {(chatHistory.length > 200 && !showAll ? chatHistory.slice(-200) : chatHistory).map((item) => {
              const isSelected = activeHistoryId === item.id;
              const isThinking = item.isThinking;
              const showLogs = expandedAccordions[item.id] !== undefined ? expandedAccordions[item.id] : isThinking;

              return (
                <div key={item.id} className="msg-group fade-in" onClick={() => handleSelectHistory(item)}>
                  {/* User bubble */}
                  <div className="msg-user">
                    <div className={`msg-user-bubble ${isSelected ? "selected" : ""}`}>
                      <div className="msg-user-meta">
                        <User size={10} />
                        <span>{item.timestamp}</span>
                      </div>
                      {item.prompt}
                    </div>
                  </div>

                  {/* AI bubble */}
                  <div className="msg-ai">
                    <div className={`msg-ai-bubble ${isSelected ? "selected" : ""}`}>
                      {/* Thinking toggle */}
                      <div
                        className={`thinking-header ${showLogs ? "open" : ""}`}
                        onClick={(e) => toggleAccordion(item.id, e)}
                      >
                        <div className="thinking-label" style={{ color: isThinking ? "var(--cyan)" : "var(--t2)" }}>
                          {isThinking
                            ? <RefreshCw size={13} className="spinning" />
                            : <Terminal size={13} />
                          }
                          <span>{isThinking ? "Thinking…" : "View execution logs"}</span>
                        </div>
                        <div className="thinking-actions">
                          {!isThinking && item.code && (
                            <button
                              className="icon-btn"
                              style={{ width: "auto", height: "auto", padding: "3px 8px", fontSize: "0.65rem", gap: "4px", display: "flex", alignItems: "center", color: "var(--cyan)", background: "var(--cyan-dim)", borderColor: "var(--cyan-border)", borderWidth: 1, borderStyle: "solid", borderRadius: "var(--r-sm)" }}
                              onClick={(e) => {
                                e.stopPropagation();
                                setReplyContext({ id: item.id, prompt: item.prompt, code: item.code });
                                logSystemEvent(`Reply context set for ${item.id}`, "UI");
                              }}
                            >
                              <MessageSquareReply size={11} />
                              Reply
                            </button>
                          )}
                          {showLogs ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                        </div>
                      </div>

                      {/* Log content */}
                      {showLogs && (
                        <div className="thinking-logs">
                          {item.logs?.length === 0
                            ? <span style={{ color: "var(--t3)", fontStyle: "italic" }}>Handshaking…</span>
                            : item.logs.map((log, li) => {
                                let col = "var(--t2)";
                                if (log.type === "system") col = "#a78bfa";
                                else if (log.type === "warn" || log.type === "error") col = "var(--red)";
                                else if (log.type === "info") col = "var(--cyan)";
                                return (
                                  <div key={li} style={{ color: col, wordBreak: "break-word" }}>
                                    <span style={{ opacity: 0.4 }}>[{log.time}] </span>{log.msg}
                                  </div>
                                );
                              })
                          }
                        </div>
                      )}

                      {/* Result footer */}
                      {!isThinking && (
                        <div className="msg-result-footer">
                          {item.path?.length > 0 ? (
                            <span style={{ color: "var(--cyan)", fontSize: "0.7rem" }}>
                              ⬡ {item.path.length} cells compiled
                            </span>
                          ) : (
                            <span style={{ color: "var(--red)", fontSize: "0.7rem" }}>
                              ⚠ No execution path found
                            </span>
                          )}
                          <span style={{ color: "var(--t3)", fontSize: "0.65rem" }}>
                            {item.code ? "✓ CODE READY" : "EMPTY"}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </>
        )}

        {/* Suggestions (only when empty) */}
        {chatHistory.length === 0 && (
          <div className="suggestions-section">
            <div className="suggestions-label">
              <HelpCircle size={11} />
              Try these examples
            </div>
            {SUGGESTIONS.map((s) => (
              <button key={s} className="suggestion-chip" onClick={() => handleRun(s)}>
                {s}
              </button>
            ))}
          </div>
        )}

        <div ref={historyEndRef} />
      </div>

      {/* Input Area */}
      <div className="chat-input-area">
        {replyContext && (
          <div className="reply-banner">
            <div className="reply-banner-text">
              <MessageSquareReply size={13} color="var(--cyan)" />
              <span>Replying to: <em>"{replyContext.prompt.substring(0, 45)}{replyContext.prompt.length > 45 ? "…" : ""}"</em></span>
            </div>
            <button className="icon-btn" style={{ width: 24, height: 24 }} onClick={() => setReplyContext(null)}>
              <X size={12} />
            </button>
          </div>
        )}
        <div className="chat-input-row">
          <textarea
            ref={textareaRef}
            className="chat-textarea"
            value={prompt}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Describe your data pipeline…"
            rows={1}
            autoFocus
          />
          <button
            className="send-btn"
            onClick={() => handleRun()}
            disabled={!prompt.trim() || localThinkingId !== null}
            title="Send (Enter)"
          >
            <Send size={17} />
          </button>
        </div>
        <div style={{ fontSize: "0.62rem", color: "var(--t3)", textAlign: "right" }}>
          Enter to send · Shift+Enter for newline
        </div>
      </div>
    </aside>
  );
}
