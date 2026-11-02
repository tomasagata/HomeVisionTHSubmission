import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The backend has no CORS, so the browser talks to it through this proxy instead
const proxy = {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/api/, ''),
  },
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: { proxy },
  preview: { proxy },
})
