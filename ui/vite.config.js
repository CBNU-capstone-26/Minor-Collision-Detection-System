import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // WSL2 등에서 Windows 브라우저가 접근할 수 있도록 0.0.0.0 바인딩
    host: true,
    // VSCode 포트 포워딩(dev tunnel)·외부 공유 시 터널 도메인 허용
    // (Vite가 모르는 Host 헤더 요청을 차단하는 것을 방지)
    allowedHosts: ['.devtunnels.ms'],
    proxy: {
      // 백엔드 FastAPI로 프록시 (개발 서버)
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
