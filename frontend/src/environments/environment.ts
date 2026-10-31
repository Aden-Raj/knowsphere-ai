export const environment = {
  // The local FastAPI process binds to IPv4 loopback (127.0.0.1).
  // Using the address directly avoids localhost resolving to an unavailable IPv6 listener.
  apiUrl: 'http://127.0.0.1:8000/api/v1',
};
