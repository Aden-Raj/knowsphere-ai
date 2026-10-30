import { HttpClient } from '@angular/common/http';
import { Component, inject, signal } from '@angular/core';
import { environment } from '../environments/environment';

@Component({
  selector: 'app-root',
  imports: [],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  private readonly http = inject(HttpClient);
  protected readonly apiStatus = signal('Checking…');

  constructor() {
    this.http.get(`${environment.apiUrl}/health`).subscribe({
      next: () => this.apiStatus.set('Connected'),
      error: () => this.apiStatus.set('Unavailable'),
    });
  }
}
