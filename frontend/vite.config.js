import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    tailwindcss(),
  ],
  server: {
    proxy: {
      // anything starting with /api will be proxied to staging
      '/api': {
        target: 'https://staging-api.pulsepro.ai',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api/, ''), // strip "/api" before forwarding
      },
    },
  },
})
