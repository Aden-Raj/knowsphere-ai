import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { parseChatAnswer } from './chat-answer-format.ts';

test('renders heading lines as heading blocks', () => {
  assert.deepEqual(parseChatAnswer('## Technical skills'), [
    { kind: 'heading', level: 2, text: 'Technical skills' },
  ]);
});

test('groups Markdown bullet lines into a list block', () => {
  assert.deepEqual(parseChatAnswer('- Python\n- FastAPI'), [
    { kind: 'unordered-list', items: ['Python', 'FastAPI'] },
  ]);
});

test('groups numbered lines into an ordered list block', () => {
  assert.deepEqual(parseChatAnswer('1. Design\n2) Implement'), [
    { kind: 'ordered-list', items: ['Design', 'Implement'] },
  ]);
});

test('preserves paragraph boundaries and explicit line breaks', () => {
  assert.deepEqual(parseChatAnswer('First line\ncontinued.\n\nSecond paragraph.'), [
    { kind: 'paragraph', text: 'First line\ncontinued.' },
    { kind: 'paragraph', text: 'Second paragraph.' },
  ]);
});

test('keeps legacy plain-text answers readable', () => {
  assert.deepEqual(parseChatAnswer('A concise legacy answer.'), [
    { kind: 'paragraph', text: 'A concise legacy answer.' },
  ]);
});

test('keeps HTML-looking content as text and the template uses safe interpolation', () => {
  const htmlLooking = '<img src=x onerror=alert(1)>';
  assert.deepEqual(parseChatAnswer(htmlLooking), [{ kind: 'paragraph', text: htmlLooking }]);

  const template = readFileSync(new URL('./chat.component.html', import.meta.url), 'utf8');
  assert.match(template, /\{\{\s*block\.text\s*\}\}/);
  assert.doesNotMatch(template, /\[innerHTML\]|\.innerHTML\s*=/i);
});

test('keeps the existing citation component in the chat template', () => {
  const template = readFileSync(new URL('./chat.component.html', import.meta.url), 'utf8');
  assert.match(template, /<app-citation-list\s+\[citations\]="message\.citations"\s*\/>/);
});
