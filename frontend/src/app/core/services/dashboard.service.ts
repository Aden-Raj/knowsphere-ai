import { inject, Injectable } from '@angular/core';
import { forkJoin } from 'rxjs';
import { ConversationsService } from './conversations.service';
import { DocumentsService } from './documents.service';

@Injectable({ providedIn: 'root' })
export class DashboardService {
  private readonly documents = inject(DocumentsService);
  private readonly conversations = inject(ConversationsService);

  load() {
    return forkJoin({ documents: this.documents.listAll(), conversations: this.conversations.list(8) });
  }
}
