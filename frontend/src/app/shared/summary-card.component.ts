import { Component, input } from '@angular/core';
import { IconComponent } from './icon.component';

@Component({
  selector: 'app-summary-card',
  standalone: true,
  imports: [IconComponent],
  template: `
    <article class="summary-card">
      <div class="summary-label">{{ label() }}<span class="summary-icon" [class]="'summary-icon ' + tone()"><app-icon [name]="icon()" /></span></div>
      <strong>{{ value() }}</strong>
      <small>{{ caption() }}</small>
    </article>
  `,
  styles: [`
    .summary-card{min-height:128px;padding:16px 18px;border:1px solid var(--line);border-radius:11px;background:linear-gradient(145deg,var(--surface-base),var(--surface-deep))}
    .summary-label{display:flex;align-items:center;justify-content:space-between;color:var(--muted);font-size:11px}
    .summary-icon{display:grid;place-items:center;width:28px;height:28px;border:1px solid var(--line-strong);border-radius:8px;background:var(--panel-raised);color:var(--accent)}
    .summary-icon app-icon{width:15px;height:15px}.summary-icon.green{border-color:var(--accent-border);background:var(--panel-raised);color:var(--accent-strong)}
    .summary-icon.amber{border-color:var(--warning-border);background:var(--panel-raised);color:var(--warning)}.summary-icon.red{border-color:var(--danger-border);background:var(--panel-raised);color:var(--danger)}
    .summary-card>strong{display:block;margin-top:12px;color:var(--text-strong);font-size:27px;font-weight:640;letter-spacing:-.9px}
    .summary-card>small{display:block;margin-top:3px;color:var(--subtle);font-size:10px}
    @media(max-width:560px){.summary-card{min-height:112px;padding:14px}}
  `],
})
export class SummaryCardComponent {
  readonly label = input.required<string>();
  readonly value = input<string | number>('—');
  readonly caption = input.required<string>();
  readonly icon = input.required<string>();
  readonly tone = input<'green' | 'amber' | 'red' | 'neutral'>('neutral');
}
