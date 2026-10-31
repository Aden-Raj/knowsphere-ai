import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    pathMatch: 'full',
    loadComponent: () => import('./pages/dashboard/dashboard.component').then((module) => module.DashboardComponent),
  },
  {
    path: 'knowledge-base',
    loadComponent: () => import('./pages/knowledge-base/knowledge-base.component').then((module) => module.KnowledgeBaseComponent),
  },
  {
    path: 'knowledge-base/:documentId',
    loadComponent: () => import('./pages/knowledge-base/document-detail.component').then((module) => module.DocumentDetailComponent),
  },
  {
    path: 'chat',
    loadComponent: () => import('./pages/chat/chat.component').then((module) => module.ChatComponent),
  },
  {
    path: 'chat/:conversationId',
    loadComponent: () => import('./pages/chat/chat.component').then((module) => module.ChatComponent),
  },
  {
    path: 'conversations',
    loadComponent: () => import('./pages/conversations/conversations.component').then((module) => module.ConversationsComponent),
  },
  {
    path: 'settings',
    loadComponent: () => import('./pages/settings/settings.component').then((module) => module.SettingsComponent),
  },
  { path: '**', redirectTo: '' },
];
