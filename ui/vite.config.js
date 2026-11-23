import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // WSL2 등에서 Windows 브라우저가 접근할 수 있도록 0.0.0.0에 엽니다.
    host: true,
    // VSCode 포트 포워딩(dev tunnel)이나 외부 공유 때 터널 도메인을 허용합니다.
    // Vite가 모르는 Host 헤더 요청을 막지 않게 하려는 것입니다.
    allowedHosts: ['.devtunnels.ms'],
    proxy: {
      // 개발 서버에서 백엔드 FastAPI로 프록시합니다.
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
