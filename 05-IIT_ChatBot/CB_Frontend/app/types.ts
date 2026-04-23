export interface SlotField {
  slot: string;
  label: string;
  kind: "boolean" | "text" | "number" | "select";
  required?: boolean;
  placeholder?: string;
  options?: { label: string; value: string }[];
}

export interface InteractionPayload {
  type: "slot_form";
  policy_id?: string;
  submit_label?: string;
  fields: SlotField[];
}

export interface ChatResponse {
  answer_markdown: string;
  message_id: string;
  conversation_id: string;
  mode: "retrieval" | "rules" | "clarify" | "clarify_topic" | "help" | "refuse" | "error";
  topic: string | null;
  topic_confidence: number;
  intent: string;
  retrieved_count: number;
  decision: Record<string, unknown> | null;
  memory: Record<string, unknown>;
  clarifying_questions: string[];
  interaction?: InteractionPayload | null;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  mode?: string;
  topic?: string | null;
  feedback?: "up" | "down" | null;
  feedbackComment?: string;
  interaction?: InteractionPayload | null;
}