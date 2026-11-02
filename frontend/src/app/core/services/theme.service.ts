import { DOCUMENT } from '@angular/common';
import { Inject, Injectable, signal } from '@angular/core';

export type Theme = 'dark' | 'light';

const THEME_STORAGE_KEY = 'knowsphere-theme';

@Injectable({ providedIn: 'root' })
export class ThemeService {
  readonly activeTheme = signal<Theme>('dark');

  constructor(@Inject(DOCUMENT) private readonly document: Document) {
    const theme = this.readStoredTheme();
    this.activeTheme.set(theme);
    this.applyTheme(theme);
  }

  setTheme(theme: Theme): void {
    if (theme !== 'light' && theme !== 'dark') return;

    this.activeTheme.set(theme);
    this.applyTheme(theme);
    try {
      this.document.defaultView?.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // The in-memory theme still works when browser storage is unavailable.
    }
  }

  private readStoredTheme(): Theme {
    try {
      const stored = this.document.defaultView?.localStorage.getItem(THEME_STORAGE_KEY);
      return stored === 'light' || stored === 'dark' ? stored : 'dark';
    } catch {
      return 'dark';
    }
  }

  private applyTheme(theme: Theme): void {
    this.document.documentElement.dataset['theme'] = theme;
    this.document.documentElement.style.colorScheme = theme;
    this.document.querySelector('meta[name="theme-color"]')?.setAttribute(
      'content', theme === 'light' ? '#f4f7f5' : '#0b1012',
    );
  }
}
