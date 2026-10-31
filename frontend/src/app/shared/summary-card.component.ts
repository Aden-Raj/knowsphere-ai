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
    .summary-card{min-height:128px;padding:16px 18px;border:1px solid var(--line);border-radius:11px;background:linear-gradient(145deg,#141e21,#101719)}
    .summary-label{display:flex;align-items:center;justify-content:space-between;color:#9aa9a5;font-size:11px}
    .summary-icon{display:grid;place-items:center;width:28px;height:28px;border:1px solid #324044;border-radius:8px;background:#1a2529;color:#95aaa3}
    .summary-icon app-icon{width:15px;height:15px}.summary-icon.green{border-color:#315344;background:#1b3028;color:#8cdbb5}
    .summary-icon.amber{border-color:#554832;background:#2c261b;color:#e2bb74}.summary-icon.red{border-color:#50343a;background:#2a1d21;color:#e3939b}
    .summary-card>strong{display:block;margin-top:12px;color:#ecf4f1;font-size:27px;font-weight:640;letter-spacing:-.9px}
    .summary-card>small{display:block;margin-top:3px;color:#72817e;font-size:10px}
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
