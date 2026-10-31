import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { catchError, combineLatest, map, of } from 'rxjs';
import { environment } from '../../../environments/environment';
import { ReadinessResponse } from '../models';

export interface BackendHealth {
  api: 'connected' | 'unavailable';
  database: 'connected' | 'unavailable' | 'unknown';
}

@Injectable({ providedIn: 'root' })
export class HealthService {
  private readonly http = inject(HttpClient);

  check() {
    const api = this.http.get<ReadinessResponse>(`${environment.apiUrl}/health`).pipe(
      map(() => 'connected' as const),
      catchError(() => of('unavailable' as const)),
    );
    const database = this.http.get<ReadinessResponse>(`${environment.apiUrl}/ready`).pipe(
      map((response) => response.database === 'connected' ? 'connected' as const : 'unknown' as const),
      catchError(() => of('unavailable' as const)),
    );
    return combineLatest([api, database]).pipe(
      map(([apiStatus, databaseStatus]) => ({ api: apiStatus, database: databaseStatus })),
    );
  }
}
