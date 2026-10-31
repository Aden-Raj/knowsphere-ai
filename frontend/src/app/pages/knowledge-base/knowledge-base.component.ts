import { DatePipe } from '@angular/common';
import { Component, inject, OnDestroy, OnInit, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { Subscription, forkJoin, interval, startWith, switchMap, takeWhile, tap } from 'rxjs';
import { apiErrorMessage } from '../../core/api-error';
import { DocumentRecord, UploadFailure } from '../../core/models';
import { DocumentsService } from '../../core/services/documents.service';
import { IconComponent } from '../../shared/icon.component';
import { StatusViewComponent } from '../../shared/status-view.component';

@Component({
  selector: 'app-knowledge-base',
  standalone: true,
  imports: [DatePipe, RouterLink, IconComponent, StatusViewComponent],
  templateUrl: './knowledge-base.component.html',
  styleUrl: './knowledge-base.component.scss',
})
export class KnowledgeBaseComponent implements OnInit, OnDestroy {
  private readonly documentsService = inject(DocumentsService);
  private polling?: Subscription;
  protected readonly documents = signal<DocumentRecord[]>([]);
  protected readonly selectedFiles = signal<File[]>([]);
  protected readonly rejected = signal<UploadFailure[]>([]);
  protected readonly reindexingIds = signal<Set<string>>(new Set());
  protected readonly loading = signal(true);
  protected readonly uploading = signal(false);
  protected readonly dragOver = signal(false);
  protected readonly error = signal('');
  protected readonly notice = signal('');

  ngOnInit(): void { this.load(); }
  ngOnDestroy(): void { this.polling?.unsubscribe(); }

  protected load(): void {
    this.loading.set(true);
    this.error.set('');
    this.documentsService.listAll().subscribe({
      next: (documents) => { this.documents.set(documents); this.loading.set(false); },
      error: (error: unknown) => { this.error.set(apiErrorMessage(error)); this.loading.set(false); },
    });
  }

  protected onFilePicker(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files) this.addFiles(Array.from(input.files));
    input.value = '';
  }

  protected onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragOver.set(false);
    if (event.dataTransfer?.files) this.addFiles(Array.from(event.dataTransfer.files));
  }

  protected addFiles(files: File[]): void {
    const accepted = files.filter((file) => /\.(pdf|docx|txt|md|markdown)$/i.test(file.name));
    const rejected = files.filter((file) => !accepted.includes(file));
    this.selectedFiles.update((current) => [...current, ...accepted]);
    if (rejected.length) this.error.set(`Unsupported file type: ${rejected.map((file) => file.name).join(', ')}`);
    else this.error.set('');
  }

  protected removeFile(index: number): void {
    this.selectedFiles.update((files) => files.filter((_, itemIndex) => itemIndex !== index));
  }

  protected upload(): void {
    if (!this.selectedFiles().length || this.uploading()) return;
    this.uploading.set(true);
    this.error.set('');
    this.notice.set('');
    this.rejected.set([]);
    this.documentsService.upload(this.selectedFiles()).subscribe({
      next: (result) => {
        this.selectedFiles.set([]);
        this.rejected.set(result.rejected);
        this.documents.update((documents) => [...result.accepted, ...documents]);
        this.uploading.set(false);
        if (result.accepted.length) {
          this.notice.set(`${result.accepted.length} file${result.accepted.length === 1 ? '' : 's'} accepted. Processing status will update as indexing finishes.`);
          this.watchProcessing(result.accepted.map((document) => document.id));
        } else this.notice.set('No files were accepted. Review the details below.');
      },
      error: (error: unknown) => {
        this.error.set(apiErrorMessage(error, 'Upload failed.'));
        this.uploading.set(false);
      },
    });
  }

  protected reindex(document: DocumentRecord): void {
    this.reindexingIds.update((ids) => new Set(ids).add(document.id));
    this.error.set('');
    this.notice.set(`Reindexing ${document.original_filename || document.title}…`);
    this.documentsService.reindex(document.id).subscribe({
      next: () => {
        this.reindexingIds.update((ids) => { const next = new Set(ids); next.delete(document.id); return next; });
        this.setStatus(document.id, 'uploaded');
        this.watchProcessing([document.id]);
      },
      error: (error: unknown) => {
        this.reindexingIds.update((ids) => { const next = new Set(ids); next.delete(document.id); return next; });
        this.error.set(apiErrorMessage(error, 'Could not queue document reindexing.'));
      },
    });
  }

  protected delete(document: DocumentRecord): void {
    const filename = document.original_filename || document.title;
    if (!window.confirm(`Delete “${filename}” and its indexed content? This cannot be undone.`)) return;
    this.documentsService.delete(document.id).subscribe({
      next: () => {
        this.documents.update((items) => items.filter((item) => item.id !== document.id));
        this.notice.set(`${filename} was deleted.`);
        this.error.set('');
      },
      error: (error: unknown) => this.error.set(apiErrorMessage(error, 'Could not delete this document.')),
    });
  }

  protected formatBytes(bytes: number | null): string {
    if (bytes === null) return '—';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  private setStatus(documentId: string, status: DocumentRecord['status']): void {
    this.documents.update((items) => items.map((document) => document.id === documentId ? { ...document, status } : document));
  }

  private watchProcessing(documentIds: string[]): void {
    this.polling?.unsubscribe();
    this.polling = interval(1800).pipe(
      startWith(0),
      switchMap(() => forkJoin(documentIds.map((id) => this.documentsService.get(id)))),
      tap((updates) => this.documents.update((current) => current.map((document) => updates.find((item) => item.id === document.id) ?? document))),
      takeWhile((updates) => updates.some((document) => document.status === 'uploaded' || document.status === 'processing'), true),
    ).subscribe({
      next: (updates) => {
        if (updates.every((document) => document.status === 'completed')) this.notice.set(`${updates.length} document${updates.length === 1 ? '' : 's'} finished processing.`);
        if (updates.some((document) => document.status === 'failed')) this.notice.set('Some documents failed to process. Review their status and reindex when ready.');
      },
      error: (error: unknown) => this.error.set(apiErrorMessage(error, 'Could not refresh document processing status.')),
    });
  }
}
