import { Component, input, output } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { IconComponent } from '../shared/icon.component';

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [RouterLink, RouterLinkActive, IconComponent],
  template: `
    @if (mobileOpen()) { <button class="scrim" type="button" aria-label="Close navigation" (click)="close.emit()"></button> }
    <aside class="sidebar" [class.mobile-open]="mobileOpen()" aria-label="Workspace navigation">
      <a class="brand" routerLink="/" (click)="close.emit()" aria-label="KnowSphere AI dashboard">
        <span class="brand-symbol"><app-icon name="spark" /></span>
        <span class="brand-copy"><strong>KnowSphere</strong><small>KNOWLEDGE INTELLIGENCE</small></span>
      </a>
      <div class="workspace-chip"><span class="workspace-avatar">N</span><span><strong>My workspace</strong><small>Personal workspace</small></span><span class="workspace-caret">⌄</span></div>
      <div class="nav-label">WORKSPACE</div>
      <nav class="nav-list" aria-label="Main navigation">
        <a routerLink="/" [routerLinkActiveOptions]="{ exact: true }" routerLinkActive="active" (click)="close.emit()"><app-icon name="grid" /><span>Overview</span></a>
        <a routerLink="/knowledge-base" routerLinkActive="active" (click)="close.emit()"><app-icon name="files" /><span>Knowledge base</span><span class="nav-count">KB</span></a>
        <a routerLink="/chat" routerLinkActive="active" (click)="close.emit()"><app-icon name="chat" /><span>AI chat</span></a>
        <a routerLink="/conversations" routerLinkActive="active" (click)="close.emit()"><app-icon name="history" /><span>Conversations</span></a>
      </nav>
      <div class="sidebar-bottom">
        <div class="nav-label">PREFERENCES</div>
        <a routerLink="/settings" routerLinkActive="active" (click)="close.emit()"><app-icon name="settings" /><span>Settings</span></a>
        <div class="sidebar-divider"></div>
        <div class="profile"><div class="profile-avatar">KS</div><div><strong>Local workspace</strong><small>KnowSphere AI</small></div><span class="profile-status" title="Local session"></span></div>
      </div>
    </aside>
  `,
  styleUrl: './sidebar.component.scss',
})
export class SidebarComponent {
  readonly mobileOpen = input(false);
  readonly close = output<void>();
}
