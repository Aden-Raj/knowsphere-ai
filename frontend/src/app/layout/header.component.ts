import { Component, output } from '@angular/core';
import { IconComponent } from '../shared/icon.component';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [IconComponent],
  template: `
    <header class="topbar">
      <button type="button" class="menu-button" (click)="menu.emit()" aria-label="Open navigation"><app-icon name="menu" /></button>
      <div class="crumb"><span class="crumb-kicker">KNOWSPHERE</span><span class="crumb-slash">/</span><span>Intelligence workspace</span></div>
      <div class="top-actions"><span class="local-badge"><i></i> Local workspace</span></div>
    </header>
  `,
  styles: [`
    .topbar{position:sticky;top:0;z-index:20;display:flex;align-items:center;justify-content:space-between;height:66px;padding:0 38px;border-bottom:1px solid var(--line);background:var(--header-bg);backdrop-filter:blur(12px)}
    .crumb{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:11px}.crumb-kicker{color:var(--subtle);font-size:9px;font-weight:700;letter-spacing:1.2px}.crumb-slash{color:var(--faint)}.top-actions{display:flex;align-items:center;gap:17px}.local-badge{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:10px}.local-badge i{width:6px;height:6px;border-radius:50%;background:var(--success)}.menu-button{display:none;border:0;background:none;color:var(--text-secondary)}
    @media(max-width:860px){.topbar{height:58px;padding:0 20px}.menu-button{display:grid;place-items:center;width:32px;height:32px;margin-right:12px}.crumb{margin-right:auto}.top-actions{gap:10px}}
  `],
})
export class HeaderComponent {
  readonly menu = output<void>();
}
