import { DatePipe, DecimalPipe } from '@angular/common';
import { Component, inject, OnInit, signal } from '@angular/core';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { apiErrorMessage } from '../../core/api-error';
import { DocumentRecord } from '../../core/models';
import { DocumentsService } from '../../core/services/documents.service';
import { IconComponent } from '../../shared/icon.component';
import { StatusViewComponent } from '../../shared/status-view.component';

@Component({
  selector: 'app-document-detail',
  standalone: true,
  imports: [DatePipe, DecimalPipe, RouterLink, IconComponent, StatusViewComponent],
  template: `
    <section class="page">
      <a class="back-link" routerLink="/knowledge-base"><app-icon name="arrow" /> Back to knowledge base</a>
      @if (loading()) { <app-status-view message="Loading document details…" /> }
      @else if (error()) { <app-status-view kind="error" [message]="error()" /> }
      @else if (document(); as item) {
        <div class="page-heading"><div><p class="eyebrow">DOCUMENT DETAILS</p><h1>{{ item.original_filename || item.title }}</h1><p>Metadata and indexing status for this source.</p></div><span class="status-badge" [class]="'status-badge ' + item.status"><i></i>{{ item.status }}</span></div>
        <section class="panel detail-panel"><div class="panel-heading"><div><h2>Source information</h2><p>Document metadata supplied by the backend.</p></div><span class="file-mark"><app-icon name="file" /></span></div><div class="detail-grid">
          <div><small>Filename</small><strong>{{ item.original_filename || item.title }}</strong></div><div><small>File type</small><strong>{{ item.mime_type || item.file_extension || 'Unavailable' }}</strong></div>
          <div><small>File size</small><strong>{{ item.file_size_bytes === null ? 'Unavailable' : (item.file_size_bytes | number) + ' bytes' }}</strong></div><div><small>Chunk count</small><strong>{{ item.chunk_count }}</strong></div>
          <div><small>Embedding model</small><strong>{{ item.embedding_model || 'Not available' }}</strong></div><div><small>Uploaded</small><strong>{{ item.created_at | date:'medium' }}</strong></div>
          <div><small>Last updated</small><strong>{{ item.updated_at | date:'medium' }}</strong></div><div><small>Processing completed</small><strong>{{ item.processing_completed_at ? (item.processing_completed_at | date:'medium') : 'Not completed' }}</strong></div>
        </div>@if (item.error_message) { <div class="error-note"><app-icon name="alert" /><span><strong>Processing error</strong><br>{{ item.error_message }}</span></div> }</section>
      }
    </section>
  `,
  styles: [`
    .back-link{display:inline-flex;align-items:center;gap:8px;color:var(--accent-strong);font-size:11px}.back-link app-icon{width:14px;height:14px;transform:rotate(180deg)}
    .status-badge{display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border:1px solid var(--accent-border);border-radius:20px;background:var(--surface-base);color:var(--accent-strong);font-size:10px;text-transform:capitalize}.status-badge i{width:5px;height:5px;border-radius:50%;background:var(--accent-surface)}.status-badge.processing,.status-badge.uploaded{border-color:var(--warning-border);background:var(--warning-bg);color:var(--warning)}.status-badge.failed{border-color:var(--danger-border);background:var(--danger-bg);color:var(--danger)}
    .file-mark{display:grid;place-items:center;width:34px;height:34px;border:1px solid var(--accent-border);border-radius:9px;background:var(--panel-raised);color:var(--accent-strong)}.file-mark app-icon{width:17px;height:17px}.detail-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0;padding:7px 20px}.detail-grid>div{display:grid;gap:7px;padding:17px 4px;border-bottom:1px solid var(--line-soft)}.detail-grid>div:nth-last-child(-n+2){border-bottom:0}.detail-grid small{color:var(--subtle);font-size:9px;font-weight:650;letter-spacing:.6px;text-transform:uppercase}.detail-grid strong{overflow-wrap:anywhere;color:var(--text-strong);font-size:12px;font-weight:540}.error-note{display:flex;gap:10px;margin:0 20px 20px;padding:12px;border:1px solid var(--danger-border);border-radius:8px;background:var(--danger-bg);color:var(--danger);font-size:11px;line-height:1.55}.error-note app-icon{width:16px;height:16px;flex:0 0 auto}.error-note strong{font-weight:650}
    @media(max-width:600px){.detail-grid{grid-template-columns:1fr}.detail-grid>div:nth-last-child(-n+2){border-bottom:1px solid var(--line-soft)}.detail-grid>div:last-child{border-bottom:0}.page-heading{align-items:flex-start}.page-heading h1{overflow-wrap:anywhere}}
  `],
})
export class DocumentDetailComponent implements OnInit {
  private readonly route = inject(ActivatedRoute);
  private readonly service = inject(DocumentsService);
  protected readonly document = signal<DocumentRecord | null>(null);
  protected readonly loading = signal(true);
  protected readonly error = signal('');

  ngOnInit(): void {
    this.route.paramMap.subscribe((params) => {
      const id = params.get('documentId');
      if (!id) return;
      this.loading.set(true);
      this.service.get(id).subscribe({
        next: (document) => { this.document.set(document); this.loading.set(false); },
        error: (error: unknown) => { this.error.set(apiErrorMessage(error)); this.loading.set(false); },
      });
    });
  }
}
