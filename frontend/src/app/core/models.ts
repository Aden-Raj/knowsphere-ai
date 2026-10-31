export type DocumentStatus = 'uploaded' | 'processing' | 'completed' | 'failed';

export interface DocumentRecord {
  id: string;
  title: string;
  original_filename: string | null;
  file_extension: string | null;
  mime_type: string | null;
  file_size_bytes: number | null;
  status: DocumentStatus;
  error_message: string | null;
  chunk_count: number;
  embedding_model: string | null;
  created_at: string;
  updated_at: string;
  processing_completed_at: string | null;
}

export interface UploadFailure {
  filename: string | null;
  detail: string;
}

export interface MultipleUploadResponse {
  accepted: DocumentRecord[];
  rejected: UploadFailure[];
}

export interface Citation {
  document_id: string;
  filename: string;
  page_number: number | null;
  chunk_id: string;
  quoted_text: string;
}

export interface RetrievedChunk {
  document_id: string;
  filename: string;
  chunk_id: string;
  chunk_index: number;
  page_number: number | null;
  text: string;
  relevance_score: number;
  included_in_context: boolean;
}

export interface QueryRequest {
  question: string;
  document_ids?: string[];
  top_k?: number;
  conversation_id?: string;
}

export interface QueryResponse {
  conversation_id: string | null;
  answer: string | null;
  citations: Citation[];
  retrieved_chunks: RetrievedChunk[];
  provider: string | null;
  model: string | null;
  generation_status: string;
  retrieval_metadata: Record<string, unknown>;
}

export interface ChatMessageRecord {
  id: string;
  role: string;
  content: string;
  citations: Citation[];
  provider: string | null;
  model: string | null;
  generation_status: string | null;
  created_at: string;
}

export interface ConversationRecord {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessageRecord[];
}

export interface ReadinessResponse {
  status: string;
  database?: string;
}
