export type ChatAnswerBlock =
  | { kind: 'heading'; level: number; text: string }
  | { kind: 'paragraph'; text: string }
  | { kind: 'unordered-list'; items: string[] }
  | { kind: 'ordered-list'; items: string[] };

type ListKind = 'unordered-list' | 'ordered-list';

/** Parse the small Markdown subset used by RAG answers without interpreting HTML. */
export function parseChatAnswer(answer: string): ChatAnswerBlock[] {
  const blocks: ChatAnswerBlock[] = [];
  let paragraphLines: string[] = [];
  let listKind: ListKind | null = null;
  let listItems: string[] = [];

  const flushParagraph = (): void => {
    if (paragraphLines.length) {
      blocks.push({ kind: 'paragraph', text: paragraphLines.join('\n') });
      paragraphLines = [];
    }
  };

  const flushList = (): void => {
    if (listKind && listItems.length) blocks.push({ kind: listKind, items: listItems });
    listKind = null;
    listItems = [];
  };

  for (const line of answer.replace(/\r\n?/g, '\n').split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    const heading = /^(#{1,6})\s+(.+?)\s*#*\s*$/.exec(trimmed);
    if (heading) {
      flushParagraph();
      flushList();
      blocks.push({ kind: 'heading', level: heading[1].length, text: heading[2] });
      continue;
    }

    const bullet = /^\s*[-*+]\s+(.+?)\s*$/.exec(line);
    const numbered = /^\s*\d+[.)]\s+(.+?)\s*$/.exec(line);
    const currentKind: ListKind | null = bullet ? 'unordered-list' : numbered ? 'ordered-list' : null;
    if (currentKind) {
      flushParagraph();
      if (listKind && listKind !== currentKind) flushList();
      listKind = currentKind;
      listItems.push((bullet ?? numbered)![1]);
      continue;
    }

    flushList();
    paragraphLines.push(line);
  }

  flushParagraph();
  flushList();
  return blocks;
}
