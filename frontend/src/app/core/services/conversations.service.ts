import { HttpClient, HttpParams } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { EMPTY, expand, reduce } from 'rxjs';
import { environment } from '../../../environments/environment';
import { ConversationRecord } from '../models';

@Injectable({ providedIn: 'root' })
export class ConversationsService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = `${environment.apiUrl}/chat/conversations`;

  list(limit = 50, offset = 0) {
    const params = new HttpParams().set('limit', limit).set('offset', offset);
    return this.http.get<ConversationRecord[]>(this.baseUrl, { params });
  }

  listAll() {
    const pageSize = 100;
    return this.list(pageSize).pipe(
      expand((page, index) => page.length === pageSize ? this.list(pageSize, (index + 1) * pageSize) : EMPTY),
      reduce((all, page) => all.concat(page), [] as ConversationRecord[]),
    );
  }

  get(conversationId: string) {
    return this.http.get<ConversationRecord>(`${this.baseUrl}/${encodeURIComponent(conversationId)}`);
  }

  delete(conversationId: string) {
    return this.http.delete<void>(`${this.baseUrl}/${encodeURIComponent(conversationId)}`);
  }
}
