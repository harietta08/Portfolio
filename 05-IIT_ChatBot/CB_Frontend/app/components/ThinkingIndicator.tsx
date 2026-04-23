export default function ThinkingIndicator() {
  return (
    <div className="flex gap-3 animate-slide-up">
      <img
        src="/iit.svg"
        alt="IIT"
        className="w-8 h-8 rounded-full flex-shrink-0 mt-0.5"
      />
      <div
        className="bg-white border border-[#E5E5EA] rounded-2xl rounded-tl-md px-4 py-4 flex items-center gap-1.5"
        style={{ boxShadow: "0 1px 4px rgba(0,0,0,0.06)" }}
      >
        {[0, 150, 300].map((delay) => (
          <span
            key={delay}
            className="w-1.5 h-1.5 rounded-full bg-[#CC0000] opacity-40 animate-typing"
            style={{ animationDelay: `${delay}ms` }}
          />
        ))}
      </div>
    </div>
  );
}
