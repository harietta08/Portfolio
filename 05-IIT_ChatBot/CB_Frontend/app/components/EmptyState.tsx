"use client";

export default function EmptyState() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-6 text-center px-8 py-12 animate-fade-in">
      {/* IIT crest-style icon */}
      <div className="relative">
        <div
          className="w-20 h-20 rounded-2xl flex items-center justify-center"
          style={{
            background: "linear-gradient(135deg, #CC0000 0%, #A30000 100%)",
            boxShadow: "0 8px 24px rgba(204,0,0,0.25)"
          }}
        >
          <span className="font-display font-800 text-white text-2xl tracking-tight">IIT</span>
        </div>
        <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-white rounded-full border-2 border-[#E5E5EA] flex items-center justify-center text-sm">
          🎓
        </div>
      </div>

      <div>
        <h2 className="font-display font-700 text-[1.6rem] text-[#1C1C1E] tracking-tight leading-tight mb-2">
          International Student Assistant
        </h2>
        <p className="text-[14px] text-[#8E8E93] max-w-[380px] leading-relaxed font-body">
          Ask me anything about F-1 status, CPT, OPT, STEM OPT, travel signatures, enrollment requirements, and more.
        </p>
      </div>
    </div>
  );
}
