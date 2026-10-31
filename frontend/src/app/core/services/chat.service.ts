import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';
import { QueryRequest, QueryResponse } from '../models';

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = `${environment.apiUrl}/chat`;

  query(request: QueryRequest) {
    return this.http.post<QueryResponse>(`${this.baseUrl}/query`, request);
  }
}
