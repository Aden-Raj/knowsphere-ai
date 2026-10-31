import { DatePipe } from '@angular/common';
import { Component, inject, OnInit, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { apiErrorMessage } from '../../core/api-error';
import { ConversationRecord } from '../../core/models';
import { ConversationsService } from '../../core/services/conversations.service';
import { IconComponent } from '../../shared/icon.component';
import { StatusViewComponent } from '../../shared/status-view.component';

@Component({
  selector: 'app-conversations',
  standalone: true,
  imports: [DatePipe, RouterLink, IconComponent, StatusViewComponent],
  templateUrl: './conversations.component.html',
  styleUrl: './conversations.component.scss',
})
export class ConversationsComponent implements OnInit {
  private readonly service = inject(ConversationsService);
  protected readonly conversations = signal<ConversationRecord[]>([]);
  protected readonly loading = signal(true);
  protected readonly error = signal('');

  ngOnInit(): void { this.load(); }
  protected load(): void {
    this.loading.set(true);
    this.service.listAll().subscribe({
      next: (conversations) => { this.conversations.set(conversations); this.loading.set(false); this.error.set(''); },
      error: (error: unknown) => { this.error.set(apiErrorMessage(error)); this.loading.set(false); },
    });
  }

  protected delete(conversation: ConversationRecord): void {
    if (!window.confirm(`Delete the conversation “${conversation.title}”? This cannot be undone.`)) return;
    this.service.delete(conversation.id).subscribe({
      next: () => this.conversations.update((items) => items.filter((item) => item.id !== conversation.id)),
      error: (error: unknown) => this.error.set(apiErrorMessage(error, 'Could not delete this conversation.')),
    });
  }
}
