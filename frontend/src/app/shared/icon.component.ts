import { Component, input } from '@angular/core';

@Component({
  selector: 'app-icon',
  standalone: true,
  template: `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      @switch (name()) {
        @case ('grid') { <rect x="3.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="3.5" y="13.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="13.5" width="7" height="7" rx="1.5"/> }
        @case ('files') { <path d="M13 3H6a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M13 3v5h5M8 13h8M8 16h8"/><path d="M18 7h1a2 2 0 0 1 2 2v11"/> }
        @case ('chat') { <path d="M20 11.5a7.5 7.5 0 0 1-8 7.5 8.8 8.8 0 0 1-3.3-.6L4 20l1.2-3.7A7 7 0 0 1 4 12c0-4.2 3.6-7.5 8-7.5s8 3.1 8 7z"/><path d="M8 12h.01M12 12h.01M16 12h.01"/> }
        @case ('history') { <path d="M3 12a9 9 0 1 0 2.6-6.4L3 8"/><path d="M3 3v5h5M12 7v5l3 2"/> }
        @case ('settings') { <circle cx="12" cy="12" r="3"/><path d="m19.4 15 .1.1a1.7 1.7 0 0 1-2.4 2.4l-.1-.1a1.7 1.7 0 0 0-2.9 1.2v.2a1.7 1.7 0 0 1-3.4 0v-.2a1.7 1.7 0 0 0-2.9-1.2l-.1.1a1.7 1.7 0 0 1-2.4-2.4l.1-.1a1.7 1.7 0 0 0-1.2-2.9H4a1.7 1.7 0 0 1 0-3.4h.2a1.7 1.7 0 0 0 1.2-2.9l-.1-.1a1.7 1.7 0 0 1 2.4-2.4l.1.1a1.7 1.7 0 0 0 2.9-1.2V2a1.7 1.7 0 0 1 3.4 0v.2a1.7 1.7 0 0 0 2.9 1.2l.1-.1a1.7 1.7 0 0 1 2.4 2.4l-.1.1a1.7 1.7 0 0 0 1.2 2.9h.2a1.7 1.7 0 0 1 0 3.4h-.2a1.7 1.7 0 0 0-1.2 2.9z"/> }
        @case ('plus') { <path d="M12 5v14M5 12h14"/> }
        @case ('search') { <circle cx="10.8" cy="10.8" r="6.8"/><path d="m16 16 4.5 4.5"/> }
        @case ('upload') { <path d="M12 16V4m0 0L8 8m4-4 4 4"/><path d="M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4"/> }
        @case ('file') { <path d="M13 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V10z"/><path d="M13 3v7h7M8 14h8M8 17h8"/> }
        @case ('clock') { <circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/> }
        @case ('check') { <path d="m5 12 4 4L19 6"/> }
        @case ('alert') { <path d="M12 3 2.8 19h18.4L12 3z"/><path d="M12 9v4m0 3h.01"/> }
        @case ('trash') { <path d="M4 7h16M10 11v6m4-6v6M6 7l1 13h10l1-13M9 7V4h6v3"/> }
        @case ('refresh') { <path d="M20 7v5h-5M4 17v-5h5"/><path d="M5.5 9A7 7 0 0 1 17 5l3 2M4 17l3 2a7 7 0 0 0 11.5-4"/> }
        @case ('arrow') { <path d="M5 12h14m-6-6 6 6-6 6"/> }
        @case ('spark') { <path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16z"/> }
        @case ('sun') { <circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M2 12h2m16 0h2M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42"/> }
        @case ('moon') { <path d="M20.5 15.4A8.5 8.5 0 0 1 8.6 3.5 8.6 8.6 0 1 0 20.5 15.4z"/> }
        @case ('menu') { <path d="M4 6h16M4 12h16M4 18h16"/> }
        @case ('close') { <path d="m6 6 12 12M18 6 6 18"/> }
        @case ('external') { <path d="M14 4h6v6m0-6-9 9"/><path d="M18 13v5a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h5"/> }
        @default { <circle cx="12" cy="12" r="8"/> }
      }
    </svg>
  `,
  styles: [`:host{display:inline-flex;width:1.15rem;height:1.15rem;flex:0 0 auto}:host svg{width:100%;height:100%}`],
})
export class IconComponent {
  readonly name = input('');
}
