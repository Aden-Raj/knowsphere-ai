import { DatePipe } from '@angular/common';
import { Component, inject, OnInit, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { apiErrorMessage } from '../../core/api-error';
import { ConversationRecord, DocumentRecord } from '../../core/models';
import { DashboardService } from '../../core/services/dashboard.service';
import { HealthService } from '../../core/services/health.service';
import { IconComponent } from '../../shared/icon.component';
import { StatusViewComponent } from '../../shared/status-view.component';
import { SummaryCardComponent } from '../../shared/summary-card.component';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [DatePipe, RouterLink, IconComponent, StatusViewComponent, SummaryCardComponent],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.scss',
})
export class DashboardComponent implements OnInit {
  private readonly dashboard = inject(DashboardService);
  private readonly healthService = inject(HealthService);
  protected readonly documents = signal<DocumentRecord[]>([]);
  protected readonly conversations = signal<ConversationRecord[]>([]);
  protected readonly loading = signal(true);
  protected readonly error = signal('');
  protected readonly apiStatus = signal<'checking' | 'connected' | 'unavailable'>('checking');
  protected readonly dbStatus = signal<'checking' | 'connected' | 'unavailable' | 'unknown'>('checking');

  ngOnInit(): void {
    this.load();
    this.healthService.check().subscribe((health) => {
      this.apiStatus.set(health.api);
      this.dbStatus.set(health.database);
    });
  }

  protected load(): void {
    this.loading.set(true);
    this.error.set('');
    this.dashboard.load().subscribe({
      next: (data) => {
        this.documents.set(data.documents);
        this.conversations.set(data.conversations);
        this.loading.set(false);
      },
      error: (error: unknown) => {
        this.error.set(apiErrorMessage(error, 'Could not load workspace activity.'));
        this.loading.set(false);
      },
    });
  }

  protected count(status: string): number {
    return this.documents().filter((document) => document.status === status).length;
  }

  protected refreshHealth(): void {
    this.apiStatus.set('checking');
    this.dbStatus.set('checking');
    this.healthService.check().subscribe((health) => {
      this.apiStatus.set(health.api);
      this.dbStatus.set(health.database);
    });
  }
}
