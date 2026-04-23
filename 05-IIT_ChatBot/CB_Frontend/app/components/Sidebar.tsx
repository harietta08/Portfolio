"use client";
import { useState } from "react";

interface Props {
  onTopicClick: (query: string) => void;
  onReset: () => void;
}

const TOPICS = [
  {
    emoji: "📋",
    label: "CPT",
    questions: [
      "How do I apply for CPT?",
      "Am I eligible for CPT?",
      "Can I do full-time CPT?",
    ],
  },
  {
    emoji: "💼",
    label: "OPT",
    questions: [
      "How do I apply for OPT?",
      "When should I apply for OPT?",
      "What documents do I need for OPT?",
      "How long is OPT valid?",
    ],
  },
  {
    emoji: "🔬",
    label: "STEM OPT",
    questions: [
      "What is the STEM OPT extension?",
      "How do I apply for STEM OPT?",
      "What is the I-983 form?",
    ],
  },
  {
    emoji: "🏫",
    label: "On-Campus Work",
    questions: [
      "Can I work on campus during the semester?",
      "How many hours can I work on campus?",
      "Can I work 40 hours on campus in summer?",
    ],
  },
  {
    emoji: "✈️",
    label: "Travel",
    questions: [
      "How do I get a travel signature?",
      "Can I travel outside the US on F-1?",
      "What documents do I need to re-enter the US?",
    ],
  },
  {
    emoji: "📚",
    label: "RCL",
    questions: [
      "What is Reduced Course Load?",
      "How do I apply for RCL?",
      "Can I drop below full-time enrollment?",
      "What are valid reasons for RCL?",
    ],
  },
  {
    emoji: "📄",
    label: "F-1 Status",
    questions: [
      "How do I maintain my F-1 status?",
      "How do I extend my I-20?",
      "What is SEVIS and why does it matter?",
    ],
  },
  {
    emoji: "🏥",
    label: "Insurance Waiver",
    questions: [
      "How do I apply for a health insurance waiver?",
      "What is the SHIP waiver deadline?",
      "What insurance qualifies for a waiver?",
      "Where do I submit my waiver request?",
    ],
  },
  {
    emoji: "🪪",
    label: "SSN",
    questions: [
      "How do I apply for an SSN?",
      "Who is eligible for an SSN?",
      "What documents do I need for SSN?",
    ],
  },
];

export default function Sidebar({ onTopicClick, onReset }: Props) {
  const [openTopic, setOpenTopic] = useState<string | null>(null);

  const toggle = (label: string) => {
    setOpenTopic((prev) => (prev === label ? null : label));
  };

  return (
    <aside className="w-56 flex-shrink-0 bg-[#FAFAFA] border-r border-[#E5E5EA] flex flex-col overflow-y-auto">
      <div className="px-4 pt-5 pb-3 border-b border-[#E5E5EA]">
        <p className="text-[10px] font-display font-700 uppercase tracking-[0.12em] text-[#8E8E93]">
          Browse Topics
        </p>
      </div>

      <div className="flex flex-col p-2 flex-1">
        {TOPICS.map((t) => {
          const isOpen = openTopic === t.label;
          return (
            <div key={t.label}>
              <button
                onClick={() => toggle(t.label)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-[13px] font-body transition-all text-left group ${
                  isOpen
                    ? "bg-[rgba(204,0,0,0.06)] text-[#CC0000]"
                    : "text-[#48484A] hover:bg-[rgba(204,0,0,0.04)] hover:text-[#CC0000]"
                }`}
              >
                <span className="text-base w-5 text-center flex-shrink-0">{t.emoji}</span>
                <span className="font-medium flex-1">{t.label}</span>
                <svg
                  className={`w-3 h-3 flex-shrink-0 transition-transform duration-200 ${isOpen ? "rotate-90 text-[#CC0000]" : "text-[#AEAEB2]"}`}
                  viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
                >
                  <polyline points="9 18 15 12 9 6" />
                </svg>
              </button>

              {isOpen && (
                <div className="ml-4 mb-1 flex flex-col gap-0.5 border-l-2 border-[rgba(204,0,0,0.2)] pl-2">
                  {t.questions.map((q) => (
                    <button
                      key={q}
                      onClick={() => onTopicClick(q)}
                      className="text-left text-[11.5px] text-[#48484A] hover:text-[#CC0000] px-2 py-1.5 rounded-md hover:bg-[rgba(204,0,0,0.05)] transition-all leading-snug font-body"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Bottom buttons */}
      <div className="p-3 border-t border-[#E5E5EA] flex flex-col gap-2">
        <button
          onClick={onReset}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-[12.5px] text-[#CC0000] hover:bg-[rgba(204,0,0,0.06)] transition-all border border-[rgba(204,0,0,0.25)] font-body"
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="1 4 1 10 7 10" />
            <path d="M3.51 15a9 9 0 1 0 .49-4.95" />
          </svg>
          New Conversation
        </button>

        <a
          href="https://forms.cloud.microsoft/Pages/ResponsePage.aspx?id=PmWQ2gIeCEK2imIkiRvqY7JxaaHt5HpLo-XJDD6JSj1UOE1ZV1M3N0hYRUdUVFlTUTRZRVZTM0VWNy4u"
          target="_blank"
          rel="noopener noreferrer"
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-[12.5px] text-white transition-all font-body"
          style={{
            background: "linear-gradient(135deg, #CC0000 0%, #A30000 100%)",
            boxShadow: "0 2px 8px rgba(204,0,0,0.3)",
          }}
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
          Conversation Feedback
        </a>
      </div>
    </aside>
  );
}
