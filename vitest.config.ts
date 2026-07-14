import { defineConfig } from 'vitest/config'
import path from 'path'

export default defineConfig({
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './frontend'),
    },
  },
  test: {
    environment: 'jsdom',
    include: ['frontend/**/*.test.{ts,tsx}', 'electron/**/*.test.ts'],
    clearMocks: true,
  },
})
