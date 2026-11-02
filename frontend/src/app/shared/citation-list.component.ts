import { Component, input } from '@angular/core';
import { Citation } from '../core/models';
import { IconComponent } from './icon.component';

@Component({
  selector: 'app-citation-list',
  standalone: true,
  imports: [IconComponent],
  template: `
    @if (citations().length) {
      <section class="citations" aria-label="Answer sources">
        <h3><app-icon name="files" /> Sources</h3>
        <div class="citation-grid">
          @for (citation of citations(); track citation.chunk_id) {
            <article class="citation-card">
              <div class="source-title"><app-icon name="file" /><span>{{ citation.filename }}</span>
                @if (citation.page_number !== null) { <small>p. {{ citation.page_number }}</small> }
              </div>
              <blockquote>{{ citation.quoted_text }}</blockquote>
              <small class="chunk-id">Chunk {{ citation.chunk_id }}</small>
            </article>
          }
        </div>
      </section>
    }
  `,
  styles: [`
    .citations{margin-top:18px}.citations h3{display:flex;align-items:center;gap:8px;margin:0 0 10px;color:var(--text-strong);font-size:12px;font-weight:650}.citations h3 app-icon{color:var(--accent)}
    .citation-grid{display:grid;gap:9px}.citation-card{padding:12px;border:1px solid var(--line);border-radius:9px;background:var(--panel)}.source-title{display:flex;align-items:center;gap:8px;color:var(--text-strong);font-size:12px}.source-title app-icon{color:var(--accent)}.source-title small{margin-left:auto;color:var(--muted)}
    blockquote{margin:10px 0 8px;padding-left:11px;border-left:2px solid var(--accent-border);color:var(--text-secondary);font-size:12px;line-height:1.6}.chunk-id{color:var(--subtle);font-size:10px}
  `],
})
export class CitationListComponent {
  readonly citations = input<Citation[]>([]);
}
