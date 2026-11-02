import { Component, input, output } from '@angular/core';
import { IconComponent } from './icon.component';

@Component({
  selector: 'app-status-view',
  standalone: true,
  imports: [IconComponent],
  template: `
    <div class="status-view" [class.error]="kind() === 'error'" [class.success]="kind() === 'success'" role="status">
      @if (kind() === 'loading') { <span class="spinner" aria-hidden="true"></span> }
      @else { <app-icon [name]="kind() === 'error' ? 'alert' : 'check'" /> }
      <span>{{ message() }}</span>
      @if (actionLabel()) { <button type="button" class="retry" (click)="action.emit()">{{ actionLabel() }}</button> }
    </div>
  `,
  styles: [`
    .status-view{display:flex;align-items:center;gap:10px;padding:14px 16px;border:1px solid var(--line);border-radius:10px;background:var(--panel);color:var(--muted);font-size:13px}
    .status-view.error{border-color:var(--danger-border);background:var(--danger-bg);color:var(--danger)}.status-view.success{border-color:var(--accent-border);background:var(--success-bg);color:var(--accent-strong)}
    .spinner{width:15px;height:15px;border:2px solid var(--line-strong);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
    .retry{margin-left:auto;border:0;background:transparent;color:var(--accent-strong);cursor:pointer}@keyframes spin{to{transform:rotate(360deg)}}
    @media(prefers-reduced-motion:reduce){.spinner{animation-duration:2s}}
  `],
})
export class StatusViewComponent {
  readonly kind = input<'loading' | 'error' | 'success'>('loading');
  readonly message = input('');
  readonly actionLabel = input('');
  readonly action = output<void>();
}
