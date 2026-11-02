import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const component = readFileSync(new URL('./chat.component.ts', import.meta.url), 'utf8');
const mapperStart = component.indexOf('private fromStoredMessage(');
const mapperEnd = component.indexOf('\n  private scrollToLatest()', mapperStart);
assert.ok(mapperStart >= 0 && mapperEnd > mapperStart, 'stored-message mapper should exist');
const storedMessageMapper = component.slice(mapperStart, mapperEnd);

test('saved clarification messages are mapped to visible assistant content', () => {
  assert.match(storedMessageMapper, /const clarification = message\.generation_status === 'clarification'/);
  assert.match(
    storedMessageMapper,
    /content:\s*!assistant \|\| message\.generation_status === 'generated' \|\| clarification \? message\.content : null/,
  );
  assert.match(
    storedMessageMapper,
    /answerBlocks:\s*assistant && \(message\.generation_status === 'generated' \|\| clarification\) \? parseChatAnswer\(message\.content\) : \[\]/,
  );
});

test('generated answers remain visible and parsed', () => {
  assert.match(storedMessageMapper, /message\.generation_status === 'generated' \|\| clarification/);
});

test('no_evidence, failed, and other non-generated messages remain hidden as before', () => {
  assert.match(
    storedMessageMapper,
    /content:\s*!assistant \|\| message\.generation_status === 'generated' \|\| clarification \? message\.content : null/,
  );
  assert.doesNotMatch(storedMessageMapper, /generation_status === '(?:no_evidence|failed|provider_error|invalid_response)'/);
});
