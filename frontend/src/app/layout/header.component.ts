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
    .topbar{position:sticky;top:0;z-index:20;display:flex;align-items:center;justify-content:space-between;height:66px;padding:0 38px;border-bottom:1px solid var(--line);background:#0f1518f2;backdrop-filter:blur(12px)}
    .crumb{display:flex;align-items:center;gap:10px;color:#93a19f;font-size:11px}.crumb-kicker{color:#647371;font-size:9px;font-weight:700;letter-spacing:1.2px}.crumb-slash{color:#4b5958}.top-actions{display:flex;align-items:center;gap:17px}.local-badge{display:flex;align-items:center;gap:8px;color:#95a5a0;font-size:10px}.local-badge i{width:6px;height:6px;border-radius:50%;background:#54c28f}.menu-button{display:none;border:0;background:none;color:#cad4d2}
    @media(max-width:860px){.topbar{height:58px;padding:0 20px}.menu-button{display:grid;place-items:center;width:32px;height:32px;margin-right:12px}.crumb{margin-right:auto}.top-actions{gap:10px}}
  `],
})
export class HeaderComponent {
  readonly menu = output<void>();
}
