# Checkbox detection: frontend

A single React + Vite page to try the API. Drop a document image on it, and it's sent to `POST /detect` and shown with every returned box drawn on top (green = checked, blue = unchecked).

```bash
npm ci
npm run dev       # http://localhost:5173, expects the backend on http://localhost:8000
```

The page calls `/api/detect` on its own origin, and Vite proxies `/api/*` to the backend (`BACKEND_URL`, default `http://localhost:8000`, see `vite.config.ts`). That way the backend doesn't need CORS. In Docker, `vite preview` serves the production build with the same proxy.
