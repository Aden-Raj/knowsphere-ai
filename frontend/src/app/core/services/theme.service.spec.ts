import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const service = readFileSync(new URL('./theme.service.ts', import.meta.url), 'utf8');
const settings = readFileSync(new URL('../../pages/settings/settings.component.html', import.meta.url), 'utf8');
const index = readFileSync(new URL('../../../index.html', import.meta.url), 'utf8');

test('dark mode is the default when no valid preference exists', () => {
  assert.match(service, /activeTheme = signal<Theme>\('dark'\)/);
  assert.match(service, /return stored === 'light' \|\| stored === 'dark' \? stored : 'dark'/);
  assert.match(service, /catch \{\s*return 'dark';/);
});

test('light and dark selections update the reactive state and persist the preference', () => {
  assert.match(service, /setTheme\(theme: Theme\): void/);
  assert.match(service, /if \(theme !== 'light' && theme !== 'dark'\) return/);
  assert.match(service, /this\.activeTheme\.set\(theme\)/);
  assert.match(service, /setItem\(THEME_STORAGE_KEY, theme\)/);
  assert.match(service, /const THEME_STORAGE_KEY = 'knowsphere-theme'/);
});

test('service initialization restores the saved preference and applies it globally', () => {
  assert.match(service, /const theme = this\.readStoredTheme\(\)/);
  assert.match(service, /this\.applyTheme\(theme\)/);
  assert.match(service, /documentElement\.dataset\['theme'\] = theme/);
  assert.match(service, /documentElement\.style\.colorScheme = theme/);
});

test('theme-color metadata follows the selected theme', () => {
  assert.match(service, /theme === 'light' \? '#f4f7f5' : '#0b1012'/);
});

test('startup restores a valid preference before Angular paints the application', () => {
  assert.match(index, /localStorage\.getItem\('knowsphere-theme'\)/);
  assert.match(index, /theme === 'light' \|\| theme === 'dark'/);
  assert.match(index, /document\.documentElement\.dataset\.theme = theme/);
});

test('Settings exposes an accessible control that reflects the current theme', () => {
  assert.match(settings, /<h2>Appearance<\/h2>/);
  assert.match(settings, /Customize how KnowSphere AI looks\./);
  assert.match(settings, /role="radiogroup" aria-label="Appearance theme"/);
  assert.match(settings, /value="light"[^>]*\[checked\]="themeService\.activeTheme\(\) === 'light'"/);
  assert.match(settings, /value="dark"[^>]*\[checked\]="themeService\.activeTheme\(\) === 'dark'"/);
  assert.match(settings, /Light Mode/);
  assert.match(settings, /Dark Mode/);
});
