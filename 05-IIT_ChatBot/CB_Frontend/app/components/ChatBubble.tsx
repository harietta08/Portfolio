"use client";

import { useState } from "react";
import { Message } from "../types";
import SlotFormCard from "./SlotFormCard";

interface Props {
  message: Message;
  onFeedback: (id: string, vote: "up" | "down") => void;
  onFeedbackComment: (id: string, comment: string) => void;
  onSlotSubmit?: (payload: string, displayText: string) => void;
}

const SLUG_MAP: Record<string, string> = {
  "employment_opt_sevp_portal": "OPT SEVP Portal",
  "employment_opt_reporting_requirements": "OPT Reporting Requirements",
  "employment_stem_opt_application_procedures": "STEM OPT Application",
  "employment_stem_opt_i983_instructions": "STEM OPT I-983",
  "employment_stem_opt": "STEM OPT",
  "employment_on_campus": "On-Campus Employment",
  "employment_cpt": "CPT",
  "employment_opt": "OPT",
  "f1_status_enrollment_requirements": "Enrollment Requirements",
  "f1_status_reduced_course_load": "Reduced Course Load",
  "f1_status_forms_and_requests": "Forms & Requests",
  "f1_status_forms_i20_ds2019": "I-20 / DS-2019",
  "f1_status_new_student_check_in": "New Student Check-In",
  "f1_status_pre_enrollment_fee": "Pre-Enrollment Fee",
  "f1_status_processing_times": "Processing Times",
  "f1_status_change_to_f1_j1": "Change to F-1/J-1",
  "f1_status_travel": "Travel",
  "health_insurance_ship_waiver": "Insurance Waiver",
  "health_insurance_shwc_services": "SHWC Services",
  "health_insurance_student_fees": "Student Fees",
  "health_insurance_ssn": "SSN",
  "OPT_sevp_portal": "OPT SEVP Portal",
  "OPT_reporting_requirements": "OPT Reporting Requirements",
  "STEM_OPT_application_procedures": "STEM OPT Application",
  "STEM_OPT_i983_instructions": "STEM OPT I-983",
};

function cleanAnswer(md: string): string {
  if (!md) return "";
  for (const [slug, label] of Object.entries(SLUG_MAP)) {
    md = md.split(slug).join(label);
  }
  md = md.replace(/\b([a-z]+(_[a-z0-9]+)+)\b/g, (match) => match.replace(/_/g, " "));
  md = md.replace(/\n{3,}/g, "\n\n");
  return md.trim();
}

function splitSources(md: string): { main: string; sources: string } {
  const marker = "### Sources";
  if (md.includes(marker)) {
    const idx = md.indexOf(marker);
    return { main: md.slice(0, idx).trim(), sources: md.slice(idx).trim() };
  }
  return { main: md.trim(), sources: "" };
}

function escapeHtml(md: string): string {
  return md
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMarkdown(md: string): string {
  if (!md) return "";

  let safe = escapeHtml(md);

  const linkTokens: string[] = [];

  // markdown links first
  safe = safe.replace(/\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g, (_m, label, url) => {
    const token = `__LINK_TOKEN_${linkTokens.length}__`;
    linkTokens.push(
      `<a href="${url}" target="_blank" rel="noopener noreferrer" class="iit-link">${label}</a>`
    );
    return token;
  });

  // raw URLs next, so Sources links stay clickable too
  safe = safe.replace(/(^|[\s>(])((https?:\/\/[^\s<]+))/g, (_m, prefix, url) => {
    const token = `__LINK_TOKEN_${linkTokens.length}__`;
    linkTokens.push(
      `<a href="${url}" target="_blank" rel="noopener noreferrer" class="iit-link">${url}</a>`
    );
    return `${prefix}${token}`;
  });

  safe = safe
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h3>$1</h3>")
    .replace(/^- (.+)$/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>\n?)+/g, (m) => "<ul>" + m + "</ul>")
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br>");

  linkTokens.forEach((html, idx) => {
    safe = safe.replace(`__LINK_TOKEN_${idx}__`, html);
  });

  return safe;
}

export default function ChatBubble({
  message,
  onFeedback,
  onFeedbackComment,
  onSlotSubmit,
}: Props) {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showComment, setShowComment] = useState(false);
  const [comment, setComment] = useState("");

  const isUser = message.role === "user";

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleFeedback = (vote: "up" | "down") => {
    onFeedback(message.id, vote);
    if (vote === "down") setShowComment(true);
  };

  if (isUser) {
    return (
      <div className="flex justify-end gap-3 animate-slide-up">
        <div className="max-w-[72%]">
          <div
            className="bg-[#CC0000] text-white rounded-2xl rounded-tr-md px-4 py-3 text-[14px] leading-relaxed font-body whitespace-pre-wrap"
            style={{ boxShadow: "0 2px 8px rgba(204,0,0,0.25)" }}
          >
            {message.content}
          </div>
        </div>
        <div className="w-8 h-8 rounded-full bg-[#E5E5EA] flex items-center justify-center text-[13px] flex-shrink-0 mt-0.5 font-display font-700 text-[#48484A]">
          You
        </div>
      </div>
    );
  }

  const cleaned = cleanAnswer(message.content);
  const { main, sources } = splitSources(cleaned);

  return (
    <div className="flex gap-3 animate-slide-up">
      <img
        src="/iit.svg"
        alt="IIT"
        className="w-8 h-8 rounded-full flex-shrink-0 mt-0.5"
      />

      <div className="max-w-[75%] flex flex-col gap-2">
        <div
          className="bg-white rounded-2xl rounded-tl-md px-4 py-3.5 text-[14px] text-[#1C1C1E] border border-[#E5E5EA]"
          style={{ boxShadow: "0 1px 4px rgba(0,0,0,0.06)" }}
        >
          <div
            className="prose-iit"
            dangerouslySetInnerHTML={{ __html: "<p>" + renderMarkdown(main) + "</p>" }}
          />

          {message.interaction?.type === "slot_form" && onSlotSubmit && (
            <SlotFormCard
              fields={message.interaction.fields}
              submitLabel={message.interaction.submit_label}
              onSubmit={onSlotSubmit}
            />
          )}
        </div>

        {sources && (
          <div>
            <button
              onClick={() => setSourcesOpen(!sourcesOpen)}
              className="flex items-center gap-1.5 text-[11.5px] text-[#8E8E93] hover:text-[#CC0000] transition-colors font-body"
            >
              <svg
                className={`w-2.5 h-2.5 transition-transform ${sourcesOpen ? "rotate-90" : ""}`}
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
              >
                <polyline points="9 18 15 12 9 6" />
              </svg>
              View sources
            </button>
            {sourcesOpen && (
              <div
                className="mt-1.5 px-3.5 py-3 bg-[#F9F9F9] border border-[#E5E5EA] rounded-xl text-[12px] text-[#48484A] leading-relaxed animate-slide-up prose-iit"
                dangerouslySetInnerHTML={{ __html: "<p>" + renderMarkdown(sources) + "</p>" }}
              />
            )}
          </div>
        )}

        <div className="flex items-center gap-4 pl-0.5">
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 text-[11px] text-[#8E8E93] hover:text-[#1C1C1E] transition-colors font-body"
          >
            {copied ? (
              <>
                <svg className="w-3 h-3 text-green-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
                <span className="text-green-500">Copied</span>
              </>
            ) : (
              <>
                <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="9" y="9" width="13" height="13" rx="2" />
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                </svg>
                <span>Copy</span>
              </>
            )}
          </button>

          <div className="flex items-center gap-1 border-l border-[#E5E5EA] pl-4">
            <button
              onClick={() => handleFeedback("up")}
              className={`text-[13px] transition-all hover:scale-110 ${message.feedback === "up" ? "grayscale-0" : "grayscale opacity-40 hover:opacity-80"}`}
              title="Helpful"
            >
              👍
            </button>
            <button
              onClick={() => handleFeedback("down")}
              className={`text-[13px] transition-all hover:scale-110 ${message.feedback === "down" ? "grayscale-0" : "grayscale opacity-40 hover:opacity-80"}`}
              title="Not helpful"
            >
              👎
            </button>
          </div>
        </div>

        {showComment && (
          <div className="animate-slide-up">
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="What went wrong? (optional)"
              rows={2}
              className="w-full bg-white border border-[#E5E5EA] rounded-xl px-3 py-2 text-[12.5px] text-[#1C1C1E] placeholder-[#AEAEB2] outline-none focus:border-[#CC0000] focus:ring-2 focus:ring-[rgba(204,0,0,0.1)] resize-none transition-all font-body"
            />
            <div className="flex gap-2 mt-1.5">
              <button
                onClick={() => {
                  onFeedbackComment(message.id, comment);
                  setShowComment(false);
                  setComment("");
                }}
                className="text-[12px] bg-[#CC0000] text-white px-4 py-1.5 rounded-lg hover:bg-[#A30000] transition-colors font-display font-600"
              >
                Submit
              </button>
              <button
                onClick={() => setShowComment(false)}
                className="text-[12px] text-[#8E8E93] hover:text-[#1C1C1E] transition-colors font-body"
              >
                Skip
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}