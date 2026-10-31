import { HttpClient, HttpParams } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { EMPTY, expand, reduce } from 'rxjs';
import { environment } from '../../../environments/environment';
import { DocumentRecord, MultipleUploadResponse } from '../models';

@Injectable({ providedIn: 'root' })
export class DocumentsService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = `${environment.apiUrl}/documents`;

  list(offset = 0, limit = 50) {
    const params = new HttpParams().set('offset', offset).set('limit', limit);
    return this.http.get<DocumentRecord[]>(this.baseUrl, { params });
  }

  listAll() {
    const pageSize = 100;
    return this.list(0, pageSize).pipe(
      expand((page, index) => page.length === pageSize ? this.list((index + 1) * pageSize, pageSize) : EMPTY),
      reduce((all, page) => all.concat(page), [] as DocumentRecord[]),
    );
  }

  get(documentId: string) {
    return this.http.get<DocumentRecord>(`${this.baseUrl}/${encodeURIComponent(documentId)}`);
  }

  upload(files: File[]) {
    const form = new FormData();
    for (const file of files) form.append('files', file, file.name);
    return this.http.post<MultipleUploadResponse>(`${this.baseUrl}/upload/multiple`, form);
  }

  reindex(documentId: string) {
    return this.http.post<DocumentRecord>(`${this.baseUrl}/${encodeURIComponent(documentId)}/reindex`, {});
  }

  delete(documentId: string) {
    return this.http.delete<void>(`${this.baseUrl}/${encodeURIComponent(documentId)}`);
  }
}
