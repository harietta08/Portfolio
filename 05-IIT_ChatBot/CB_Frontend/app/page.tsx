"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import ChatBubble from "./components/ChatBubble";
import Sidebar from "./components/Sidebar";
import ThinkingIndicator from "./components/ThinkingIndicator";
import EmptyState from "./components/EmptyState";
import { Message, ChatResponse } from "./types";

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [memory, setMemory] = useState<Record<string, unknown>>({});
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sessionId, setSessionId] = useState<string>(() => `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = useCallback(
    async (query: string, displayOverride?: string) => {
      const q = query.trim();
      if (!q || loading) return;

      setSidebarOpen(false);
      setInput("");
      if (textareaRef.current) textareaRef.current.style.height = "auto";
      setLoading(true);

      setMessages((prev) => [
        ...prev,
        {
          id: `u-${Date.now()}`,
          role: "user",
          content: displayOverride?.trim() || q,
        },
      ]);

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: q, memory }),
        });

        const data: ChatResponse = await res.json();
        setMemory(data.memory || {});
        setMessages((prev) => [
          ...prev,
          {
            id: data.message_id || `b-${Date.now()}`,  // ← use API message_id
            role: "assistant",
            content: data.answer_markdown,
            mode: data.mode,
            topic: data.topic,
            feedback: null,
            interaction: data.interaction || null,
          },
        ]);
      } catch {
        setMessages((prev) => [
          ...prev,
          {
            id: `err-${Date.now()}`,
            role: "assistant",
            content:
              "⚠️ Could not reach the backend. Make sure your FastAPI server is running on `http://localhost:8000`.",
            mode: "error",
            topic: null,
            feedback: null,
            interaction: null,
          },
        ]);
      }

      setLoading(false);
      textareaRef.current?.focus();
    },
    [loading, memory]
  );

  const handleSlotSubmit = useCallback(
    async (payload: string, displayText: string) => {
      if (!payload.trim() || loading) return;
      await sendMessage(payload, displayText);
    },
    [loading, sendMessage]
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const handleAutoResize = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
  };

const handleFeedback = async (id: string, vote: "up" | "down") => {
  setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, feedback: vote } : m)));

  if (vote === "up") {
    const msgIndex = messages.findIndex((m) => m.id === id);
    const botMsg = messages[msgIndex];
    const userMsg = messages.slice(0, msgIndex).reverse().find((m) => m.role === "user");

    try {
      await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: sessionId,
          message_id: id,
          vote,
          query: userMsg?.content || "",
          bot_answer: botMsg?.content || "",
          comment: "",
        }),
      });
    } catch (e) {
      console.error("Feedback failed:", e);
    }
  }
};
  const handleFeedbackComment = async (id: string, comment: string) => {
  const msgIndex = messages.findIndex((m) => m.id === id);
  const botMsg = messages[msgIndex];
  const userMsg = messages.slice(0, msgIndex).reverse().find((m) => m.role === "user");

  try {
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: sessionId,    // ← persistent session ID
        message_id: id,
        vote: "down",
        query: userMsg?.content || "",
        bot_answer: botMsg?.content || "",
        comment: comment,
      }),
    });
  } catch (e) {
    console.error("Feedback comment failed:", e);
  }
};

  const resetChat = () => {
  setMessages([]);
  setMemory({});
  setSidebarOpen(false);
  setSessionId(`session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`);
};

  return (
    <div className="flex flex-col h-screen bg-[#F5F5F7] overflow-hidden">
      <header className="flex-shrink-0 bg-[#CC0000] text-white flex items-center justify-between px-4 md:px-6 h-[56px] md:h-[60px] shadow-md z-20">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="md:hidden flex flex-col gap-1 p-1"
            aria-label="Toggle topics"
          >
            <span className={`block w-5 h-0.5 bg-white transition-all ${sidebarOpen ? "rotate-45 translate-y-1.5" : ""}`} />
            <span className={`block w-5 h-0.5 bg-white transition-all ${sidebarOpen ? "opacity-0" : ""}`} />
            <span className={`block w-5 h-0.5 bg-white transition-all ${sidebarOpen ? "-rotate-45 -translate-y-1.5" : ""}`} />
          </button>

          <div className="flex items-center gap-3 border-r border-[rgba(255,255,255,0.25)] pr-3 md:pr-4">
            <span className="font-display font-800 text-xl tracking-tight leading-none">IIT</span>
            <div className="hidden sm:block">
              <div className="text-[10px] font-display font-600 uppercase tracking-[0.12em] text-[rgba(255,255,255,0.75)] leading-none">
                Illinois Institute of Technology
              </div>
            </div>
          </div>
          <div className="font-display font-600 text-[14px] md:text-[15px] leading-tight">
            International Student Assistant
          </div>
        </div>

        <div className="flex items-center gap-2 md:gap-3">
          <div className="hidden sm:flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-green-300 shadow-[0_0_4px_rgba(134,239,172,0.8)]" />
            <span className="text-[11px] text-[rgba(255,255,255,0.7)] font-body">Online</span>
          </div>
          <button
            onClick={resetChat}
            className="flex items-center gap-1.5 text-[12px] text-white border border-[rgba(255,255,255,0.3)] px-2.5 md:px-3 py-1.5 rounded-lg hover:bg-[rgba(255,255,255,0.1)] transition-all font-body"
          >
            <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="1 4 1 10 7 10" />
              <path d="M3.51 15a9 9 0 1 0 .49-4.95" />
            </svg>
            <span className="hidden sm:inline">New Chat</span>
          </button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden relative">
        {sidebarOpen && (
          <div
            className="fixed inset-0 bg-black/30 z-10 md:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        <div
          className={`
            fixed md:relative top-0 left-0 h-full z-20
            transition-transform duration-300 ease-in-out
            md:translate-x-0 md:flex-shrink-0
            ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}
          `}
          style={{ marginTop: sidebarOpen ? "56px" : "0" }}
        >
          <Sidebar onTopicClick={sendMessage} onReset={resetChat} />
        </div>

        <div className="flex flex-col flex-1 overflow-hidden min-w-0">
          <div className="flex-1 overflow-y-auto px-3 md:px-6 py-4 md:py-6 flex flex-col gap-4">
            {messages.length === 0 && !loading ? (
              <EmptyState />
            ) : (
              <>
                {messages.map((msg) => (
                  <ChatBubble
                    key={msg.id}
                    message={msg}
                    onFeedback={handleFeedback}
                    onFeedbackComment={handleFeedbackComment}
                    onSlotSubmit={handleSlotSubmit}
                  />
                ))}
                {loading && <ThinkingIndicator />}
              </>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="flex-shrink-0 bg-white border-t border-[#E5E5EA] px-3 md:px-6 py-3 md:py-4">
            <div
              className="flex items-end gap-2 md:gap-3 bg-[#F5F5F7] border border-[#E5E5EA] rounded-2xl px-3 md:px-4 py-2.5 focus-within:border-[#CC0000] focus-within:bg-white transition-all"
              style={{ boxShadow: "0 1px 3px rgba(0,0,0,0.04)" }}
            >
              <textarea
                ref={textareaRef}
                value={input}
                onChange={handleAutoResize}
                onKeyDown={handleKeyDown}
                placeholder="Ask about CPT, OPT, travel, enrollment..."
                rows={1}
                disabled={loading}
                className="flex-1 bg-transparent border-none outline-none text-[14px] text-[#1C1C1E] placeholder-[#AEAEB2] resize-none max-h-[120px] leading-relaxed py-1 font-body disabled:opacity-50"
              />
              <button
                onClick={() => sendMessage(input)}
                disabled={loading || !input.trim()}
                className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
                style={{
                  background: input.trim() ? "linear-gradient(135deg, #CC0000 0%, #A30000 100%)" : "#E5E5EA",
                  boxShadow: input.trim() ? "0 2px 8px rgba(204,0,0,0.3)" : "none",
                }}
              >
                <svg
                  className={`w-4 h-4 ${input.trim() ? "text-white" : "text-[#AEAEB2]"}`}
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              </button>
            </div>

            <p className="hidden md:block text-[11px] text-[#AEAEB2] text-center mt-2 font-body">
              Press{" "}
              <kbd className="px-1.5 py-0.5 rounded bg-[#F2F2F7] border border-[#E5E5EA] text-[10px] text-[#48484A]">Enter</kbd>
              {" "}to send ·{" "}
              <kbd className="px-1.5 py-0.5 rounded bg-[#F2F2F7] border border-[#E5E5EA] text-[10px] text-[#48484A]">Shift+Enter</kbd>
              {" "}for new line
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}