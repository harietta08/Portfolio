"use client";

import { useState } from "react";
import { SlotField } from "../types";

interface Props {
  fields: SlotField[];
  submitLabel?: string;
  onSubmit: (payload: string, displayText: string) => void;
}

type FormState = Record<string, string>;

function prettifyValue(field: SlotField, value: string): string {
  if (field.kind === "boolean") {
    return value.toLowerCase() === "yes" ? "Yes" : "No";
  }

  if (field.kind === "select" && field.options?.length) {
    const match = field.options.find((opt) => opt.value === value);
    return match?.label || value;
  }

  return value;
}

export default function SlotFormCard({ fields, submitLabel, onSubmit }: Props) {
  const [values, setValues] = useState<FormState>({});

  const setValue = (slot: string, value: string) => {
    setValues((prev) => ({ ...prev, [slot]: value }));
  };

  const isValid = fields.every((f) => {
    if (!f.required) return true;
    const v = values[f.slot];
    return v !== undefined && v !== "";
  });

  const handleSubmit = () => {
  const rawLines = fields
    .filter((f) => values[f.slot] !== undefined && values[f.slot] !== "")
    .map((f) => `${f.slot}: ${values[f.slot]}`);

  const prettyLines = fields
    .filter((f) => values[f.slot] !== undefined && values[f.slot] !== "")
    .map((f) => `${f.label}: ${prettifyValue(f, values[f.slot])}`);

  const displayText =
    prettyLines.length > 0
      ? "My answers:\n" + prettyLines.join("\n")
      : "";

  onSubmit(rawLines.join("\n"), displayText);
};

  return (
    <div className="mt-3 rounded-2xl border border-[#E5E5EA] bg-[#FAFAFA] p-4">
      <div className="flex flex-col gap-4">
        {fields.map((field) => (
          <div key={field.slot} className="flex flex-col gap-2">
            <label className="text-[13px] font-medium text-[#1C1C1E]">
              {field.label}
            </label>

            {field.kind === "boolean" && (
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setValue(field.slot, "yes")}
                  className={`px-3 py-2 rounded-xl text-[12px] border transition-all ${
                    values[field.slot] === "yes"
                      ? "bg-[#CC0000] text-white border-[#CC0000]"
                      : "bg-white text-[#1C1C1E] border-[#D1D1D6]"
                  }`}
                >
                  Yes
                </button>
                <button
                  type="button"
                  onClick={() => setValue(field.slot, "no")}
                  className={`px-3 py-2 rounded-xl text-[12px] border transition-all ${
                    values[field.slot] === "no"
                      ? "bg-[#1C1C1E] text-white border-[#1C1C1E]"
                      : "bg-white text-[#1C1C1E] border-[#D1D1D6]"
                  }`}
                >
                  No
                </button>
              </div>
            )}

            {field.kind === "text" && (
              <input
                type="text"
                value={values[field.slot] || ""}
                onChange={(e) => setValue(field.slot, e.target.value)}
                placeholder={field.placeholder || "Enter your answer"}
                className="w-full rounded-xl border border-[#D1D1D6] bg-white px-3 py-2 text-[13px] outline-none focus:border-[#CC0000]"
              />
            )}

            {field.kind === "number" && (
              <input
                type="number"
                value={values[field.slot] || ""}
                onChange={(e) => setValue(field.slot, e.target.value)}
                placeholder={field.placeholder || "Enter a number"}
                className="w-full rounded-xl border border-[#D1D1D6] bg-white px-3 py-2 text-[13px] outline-none focus:border-[#CC0000]"
              />
            )}

            {field.kind === "select" && (
              <select
                value={values[field.slot] || ""}
                onChange={(e) => setValue(field.slot, e.target.value)}
                className="w-full rounded-xl border border-[#D1D1D6] bg-white px-3 py-2 text-[13px] outline-none focus:border-[#CC0000]"
              >
                <option value="">Select an option</option>
                {(field.options || []).map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            )}
          </div>
        ))}

        <button
          type="button"
          disabled={!isValid}
          onClick={handleSubmit}
          className="self-start rounded-xl bg-[#CC0000] px-4 py-2 text-white text-[13px] font-medium disabled:opacity-40"
        >
          {submitLabel || "Submit"}
        </button>
      </div>
    </div>
  );
}