import { Component, inject, OnInit, signal } from '@angular/core';
import { environment } from '../../../environments/environment';
import { BackendHealth, HealthService } from '../../core/services/health.service';
import { IconComponent } from '../../shared/icon.component';
import { StatusViewComponent } from '../../shared/status-view.component';
import { Theme, ThemeService } from '../../core/services/theme.service';

@Component({
  selector: 'app-settings',
  standalone: true,
  imports: [IconComponent, StatusViewComponent],
  templateUrl: './settings.component.html',
  styleUrl: './settings.component.scss',
})
export class SettingsComponent implements OnInit {
  private readonly healthService = inject(HealthService);
  protected readonly themeService = inject(ThemeService);
  protected readonly apiUrl = environment.apiUrl;
  protected readonly health = signal<BackendHealth | null>(null);
  protected readonly checking = signal(true);

  ngOnInit(): void { this.check(); }
  protected setTheme(theme: Theme): void { this.themeService.setTheme(theme); }
  protected check(): void {
    this.checking.set(true);
    this.healthService.check().subscribe((health) => { this.health.set(health); this.checking.set(false); });
  }
}
