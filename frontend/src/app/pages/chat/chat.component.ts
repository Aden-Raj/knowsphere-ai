import { DatePipe, DecimalPipe } from '@angular/common';
import { Component, ElementRef, inject, OnInit, signal, ViewChild } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { apiErrorMessage } from '../../core/api-error';
import { ChatMessageRecord, ConversationRecord, QueryResponse, RetrievedChunk } from '../../core/models';
import { ChatAnswerBlock, parseChatAnswer } from './chat-answer-format';
import { ChatService } from '../../core/services/chat.service';
import { ConversationsService } from '../../core/services/conversations.service';
import { DocumentsService } from '../../core/services/documents.service';
import { CitationListComponent } from '../../shared/citation-list.component';
import { IconComponent } from '../../shared/icon.component';
import { StatusViewComponent } from '../../shared/status-view.component';

interface DisplayMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string | null;
  answerBlocks: ChatAnswerBlock[];
  citations: QueryResponse['citations'];
  evidence: RetrievedChunk[];
  generationStatus: string | null;
  createdAt: string;
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [FormsModule, DatePipe, DecimalPipe, CitationListComponent, IconComponent, StatusViewComponent],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.scss',
})
export class ChatComponent implements OnInit {
  private readonly route = inject(ActivatedRoute);
  protected readonly router = inject(Router);
  private readonly chatService = inject(ChatService);
  private readonly conversationsService = inject(ConversationsService);
  private readonly documentsService = inject(DocumentsService);
  @ViewChild('messageInput') private messageInput?: ElementRef<HTMLTextAreaElement>;
  @ViewChild('messagesViewport') private messagesViewport?: ElementRef<HTMLDivElement>;
  private followLatest = true;

  protected readonly conversations = signal<ConversationRecord[]>([]);
  protected readonly messages = signal<DisplayMessage[]>([]);
  protected readonly documents = signal<{ id: string; filename: string }[]>([]);
  protected readonly selectedDocumentIds = signal<string[]>([]);
  protected readonly currentConversationId = signal<string | null>(null);
  protected readonly draft = signal('');
  protected readonly loadingHistory = signal(false);
  protected readonly sending = signal(false);
  protected readonly error = signal('');
  protected readonly conversationError = signal('');

  ngOnInit(): void {
    this.refreshConversations();
    this.documentsService.listAll().subscribe({
      next: (documents) => this.documents.set(documents.filter((document) => document.status === 'completed').map((document) => ({ id: document.id, filename: document.original_filename || document.title }))),
      error: () => undefined,
    });
    this.route.paramMap.subscribe((params) => {
      const id = params.get('conversationId');
      if (id && id !== this.currentConversationId()) this.openConversation(id);
      else if (!id && this.currentConversationId()) this.startNewConversation(false);
    });
  }

  protected refreshConversations(): void {
    this.conversationsService.list(50).subscribe({
      next: (conversations) => { this.conversations.set(conversations); this.conversationError.set(''); },
      error: (error: unknown) => this.conversationError.set(apiErrorMessage(error, 'Could not load conversation history.')),
    });
  }

  protected startNewConversation(navigate = true): void {
    this.followLatest = true;
    this.currentConversationId.set(null);
    this.messages.set([]);
    this.error.set('');
    this.draft.set('');
    if (navigate) void this.router.navigate(['/chat']);
    queueMicrotask(() => this.messageInput?.nativeElement.focus());
  }

  protected openConversation(id: string): void {
    this.followLatest = true;
    this.currentConversationId.set(id);
    this.messages.set([]);
    this.loadingHistory.set(true);
    this.error.set('');
    this.conversationsService.get(id).subscribe({
      next: (conversation) => {
        this.messages.set(conversation.messages.map((message) => this.fromStoredMessage(message)));
        this.loadingHistory.set(false);
        this.scrollToLatest();
      },
      error: (error: unknown) => { this.error.set(apiErrorMessage(error, 'Could not open this conversation.')); this.loadingHistory.set(false); },
    });
  }

  protected send(): void {
    const question = this.draft().trim();
    if (!question || this.sending()) return;
    this.error.set('');
    this.sending.set(true);
    this.followLatest = true;
    this.messages.update((messages) => [...messages, {
      id: crypto.randomUUID(), role: 'user', content: question, citations: [], evidence: [],
      answerBlocks: [], generationStatus: null, createdAt: new Date().toISOString(),
    }]);
    this.draft.set('');
    this.scrollToLatest();
    const request: { question: string; document_ids?: string[]; top_k: number; conversation_id?: string } = { question, top_k: 5 };
    if (this.selectedDocumentIds().length) request.document_ids = this.selectedDocumentIds();
    if (this.currentConversationId()) request.conversation_id = this.currentConversationId()!;
    this.chatService.query(request).subscribe({
      next: (result) => {
        const id = result.conversation_id;
        this.messages.update((messages) => [...messages, {
          id: crypto.randomUUID(), role: 'assistant', content: result.answer,
          answerBlocks: result.answer ? parseChatAnswer(result.answer) : [],
          citations: result.citations, evidence: result.retrieved_chunks,
          generationStatus: result.generation_status, createdAt: new Date().toISOString(),
        }]);
        if (id && id !== this.currentConversationId()) {
          this.currentConversationId.set(id);
          void this.router.navigate(['/chat', id], { replaceUrl: true });
        }
        this.refreshConversations();
        this.sending.set(false);
        if (this.followLatest) this.scrollToLatest();
      },
      error: (error: unknown) => {
        this.error.set(apiErrorMessage(error, 'Your question could not be sent.'));
        this.sending.set(false);
      },
    });
  }

  protected onComposerKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.send();
    }
  }

  protected toggleDocument(documentId: string, checked: boolean): void {
    this.selectedDocumentIds.update((ids) => checked ? [...new Set([...ids, documentId])] : ids.filter((id) => id !== documentId));
  }

  protected isDocumentSelected(documentId: string): boolean {
    return this.selectedDocumentIds().includes(documentId);
  }

  protected onMessagesScroll(): void {
    const viewport = this.messagesViewport?.nativeElement;
    if (!viewport) return;
    this.followLatest = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight <= 56;
  }

  protected statusText(status: string | null): string {
    switch (status) {
      case 'provider_unavailable': return 'Answer generation is unavailable because no LLM provider is configured. Retrieved evidence is shown below.';
      case 'provider_error': return 'The configured provider could not generate an answer. Retrieved evidence is still available below.';
      case 'invalid_response': return 'The provider response could not be validated, so no answer was displayed. Retrieved evidence is shown below.';
      case 'no_evidence': return 'No relevant evidence was found for this question in the selected documents.';
      default: return 'No answer was generated. Retrieved evidence is shown below.';
    }
  }

  protected statusTitle(status: string | null): string {
    if (status === 'provider_unavailable') return 'Generation not configured';
    if (status === 'no_evidence') return 'No relevant evidence';
    if (status === 'provider_error') return 'Generation failed';
    return 'Answer unavailable';
  }

  protected asStoredMessage(message: ChatMessageRecord): DisplayMessage {
    return this.fromStoredMessage(message);
  }

  private fromStoredMessage(message: ChatMessageRecord): DisplayMessage {
    const assistant = message.role === 'assistant';
    return {
      id: message.id,
      role: assistant ? 'assistant' : 'user',
      content: !assistant || message.generation_status === 'generated' ? message.content : null,
      answerBlocks: assistant && message.generation_status === 'generated' ? parseChatAnswer(message.content) : [],
      citations: message.citations || [],
      evidence: [],
      generationStatus: message.generation_status,
      createdAt: message.created_at,
    };
  }

  private scrollToLatest(): void {
    setTimeout(() => {
      if (this.followLatest && this.messagesViewport) {
        const viewport = this.messagesViewport.nativeElement;
        viewport.scrollTop = viewport.scrollHeight;
      }
    }, 0);
  }
}
