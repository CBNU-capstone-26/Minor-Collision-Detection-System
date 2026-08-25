import { useMemo, useState } from "react";
import "./App.css";

// 1. 영상 중심의 Mock 데이터 (영상 1개 안에 여러 이벤트가 포함됨)
const mockVideos = [
  {
    id: "v1",
    date: "2026-05-24", // 오늘 날짜 기준
    camera: "지하 1층 CCTV-01",
    duration: 3600, // 총 영상 길이 (초 단위, 1시간)
    events: [
      { id: 101, timestamp: 450, title: "좌측 앞문 스크래치 의심", status: "확인 필요" },
      { id: 102, timestamp: 1820, title: "문콕 접촉 의심", status: "분석 완료" },
    ],
  },
  {
    id: "v2",
    date: "2026-05-22",
    camera: "정문 옥외 주차장",
    duration: 7200, // 2시간
    events: [
      { id: 201, timestamp: 3400, title: "범퍼 접촉 의심", status: "오탐 가능" },
    ],
  },
  {
    id: "v3",
    date: "2026-05-15",
    camera: "후문 주차장",
    duration: 1800, // 30분
    events: [
      { id: 301, timestamp: 900, title: "우측 뒷문 충돌 의심", status: "확인 필요" },
      { id: 302, timestamp: 1100, title: "사람 접근 감지", status: "분석 완료" },
      { id: 303, timestamp: 1550, title: "차량 긁힘 의심", status: "확인 필요" },
    ],
  },
  {
    id: "v4",
    date: "2026-04-10",
    camera: "지하 2층 CCTV-04",
    duration: 3600,
    events: [
      { id: 401, timestamp: 2100, title: "기둥 충돌 의심", status: "분석 완료" },
    ],
  },
];

function LoginPage({ onLogin }) {
  const [id, setId] = useState("");
  const [pw, setPw] = useState("");

  return (
    <div className="login-layout">
      <div className="login-left-panel">
        <div className="brand-content">
          <div className="brand-badge">Parking Scratch Detection</div>
          <h1>주차 사고 이벤트 확인 시스템</h1>
          <p>
            CCTV 영상에서 차량 접촉/스크래치 의심 이벤트를 감지하고
            날짜별로 정리하여 확인할 수 있는 웹 UI입니다.
          </p>
        </div>
      </div>

      <div className="login-right-panel">
        <div className="login-form-wrapper">
          <h2>로그인</h2>
          <p className="login-subtitle">관리자 화면으로 접속합니다.</p>

          <div className="input-group">
            <input
              type="text"
              placeholder="admin"
              value={id}
              onChange={(e) => setId(e.target.value)}
            />
          </div>

          <div className="input-group">
            <input
              type="password"
              placeholder="password"
              value={pw}
              onChange={(e) => setPw(e.target.value)}
            />
          </div>

          <div className="login-options">
            <label className="remember-me">
              <input type="checkbox" /> Remember me
            </label>
            <a href="#none" className="support-link">Support</a>
          </div>

          <button className="login-submit-btn" onClick={onLogin}>로그인</button>
        </div>
      </div>
    </div>
  );
}

// 유틸 함수들
function formatDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function addDays(date, days) {
  const copied = new Date(date);
  copied.setDate(copied.getDate() + days);
  return copied;
}

function formatTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

// 대시보드 컴포넌트
function Dashboard({ onLogout }) {
  const today = new Date();
  
  // 상태 관리
  const [filterDays, setFilterDays] = useState(7); // 기본 1주일
  const [selectedVideo, setSelectedVideo] = useState(null); // null이면 홈(그리드) 화면, 값이 있으면 영상 재생 화면
  const [currentEventId, setCurrentEventId] = useState(null); // 현재 선택된 이벤트 마커

  // 날짜 필터링 계산
  const filteredVideos = useMemo(() => {
    const startDate = formatDate(addDays(today, -filterDays));
    return mockVideos.filter((video) => video.date >= startDate);
  }, [filterDays, today]);

  // 영상을 클릭하여 시청 모드로 진입
  const handleWatchVideo = (video) => {
    setSelectedVideo(video);
    setCurrentEventId(null);
  };

  // 홈으로 돌아가기
  const handleBackToHome = () => {
    setSelectedVideo(null);
    setCurrentEventId(null);
  };

  return (
    <div className="dashboard-layout">
      {/* 상단 네비게이션 바 */}
      <header className="top-navbar">
        <div className="nav-left">
          <div className="nav-logo">AI COMS</div>
          <span className="nav-title">안전 모니터링 시스템</span>
        </div>
        <button className="nav-logout-btn" onClick={onLogout}>로그아웃</button>
      </header>

      {/* 컨텐츠 영역: 선택된 비디오가 없으면 유튜브 홈 스타일(그리드), 있으면 시청 스타일 */}
      {!selectedVideo ? (
        <main className="home-view">
          {/* 기간 필터 (유튜브 카테고리 필터 스타일) */}
          <div className="filter-pills">
            <button className={filterDays === 7 ? "active" : ""} onClick={() => setFilterDays(7)}>1주일</button>
            <button className={filterDays === 14 ? "active" : ""} onClick={() => setFilterDays(14)}>2주일</button>
            <button className={filterDays === 30 ? "active" : ""} onClick={() => setFilterDays(30)}>1달</button>
            <button className={filterDays === 90 ? "active" : ""} onClick={() => setFilterDays(90)}>3달</button>
            <button className={filterDays === 9999 ? "active" : ""} onClick={() => setFilterDays(9999)}>전체</button>
          </div>

          {/* 영상 썸네일 그리드 */}
          <div className="video-grid">
            {filteredVideos.length === 0 ? (
              <div className="empty-state">선택한 기간에 해당하는 영상이 없습니다.</div>
            ) : (
              filteredVideos.map((video) => (
                <div key={video.id} className="video-card" onClick={() => handleWatchVideo(video)}>
                  <div className="video-thumbnail">
                    {/* 썸네일 이미지 자리 */}
                    <span className="event-count-badge">이벤트 {video.events.length}건</span>
                  </div>
                  <div className="video-info">
                    <h3>{video.date} {video.camera} 녹화본</h3>
                    <p>영상 길이: {formatTime(video.duration)}</p>
                  </div>
                </div>
              ))
            )}
          </div>
        </main>
      ) : (
        <main className="watch-view">
          <button className="back-btn" onClick={handleBackToHome}>
            ← 목록으로 돌아가기
          </button>

          <div className="watch-layout">
            {/* 좌측: 이벤트 목록 (유튜브 추천영상 위치 대신 이벤트 목록 배치) */}
            <aside className="event-sidebar">
              <h3>감지된 이벤트 목록</h3>
              <p className="event-summary">총 {selectedVideo.events.length}건의 이벤트가 있습니다.</p>
              
              <div className="event-list">
                {selectedVideo.events.map((event) => (
                  <div 
                    key={event.id} 
                    className={`event-item ${currentEventId === event.id ? 'active' : ''}`}
                    onClick={() => setCurrentEventId(event.id)}
                  >
                    <div className="event-time">{formatTime(event.timestamp)}</div>
                    <div className="event-details">
                      <h4>{event.title}</h4>
                      <span className={`status-tag ${event.status}`}>{event.status}</span>
                    </div>
                  </div>
                ))}
              </div>
            </aside>

            {/* 우측/중앙: 메인 비디오 플레이어 */}
            <section className="player-section">
              <div className="player-container">
                {/* TODO: 실제 영상이 준비되면 아래 div 대신 video 태그를 사용하세요. 
                  <video src="실제경로.mp4" controls width="100%"></video>
                */}
                <div className="mock-video-player">
                  <div className="play-icon">▶</div>
                  <span className="video-overlay-title">{selectedVideo.camera} / {selectedVideo.date}</span>
                </div>

                {/* 커스텀 재생바 및 이벤트 마커 표시 */}
                <div className="custom-progress-bar">
                  <div className="progress-track">
                    {/* 재생 진행률 (임시로 0%로 고정) */}
                    <div className="progress-fill" style={{ width: "0%" }}></div>
                    
                    {/* 이벤트 마커 (타임라인의 빨간 점) */}
                    {selectedVideo.events.map((event) => {
                      const leftPosition = (event.timestamp / selectedVideo.duration) * 100;
                      return (
                        <div 
                          key={event.id}
                          className={`event-marker ${currentEventId === event.id ? 'active' : ''}`}
                          style={{ left: `${leftPosition}%` }}
                          title={event.title}
                          onClick={() => setCurrentEventId(event.id)}
                        />
                      );
                    })}
                  </div>
                  <div className="time-labels">
                    <span>0:00</span>
                    <span>{formatTime(selectedVideo.duration)}</span>
                  </div>
                </div>
              </div>

              <div className="video-metadata">
                <h2>{selectedVideo.date} {selectedVideo.camera} 전체 녹화본</h2>
                <div className="metadata-row">
                  <span className="metadata-item"><strong>카메라:</strong> {selectedVideo.camera}</span>
                  <span className="metadata-item"><strong>날짜:</strong> {selectedVideo.date}</span>
                  <span className="metadata-item"><strong>총 이벤트:</strong> {selectedVideo.events.length}건 감지됨</span>
                </div>
              </div>
            </section>
          </div>
        </main>
      )}
    </div>
  );
}

export default function App() {
  const [isLogin, setIsLogin] = useState(false);

  return isLogin ? (
    <Dashboard onLogout={() => setIsLogin(false)} />
  ) : (
    <LoginPage onLogin={() => setIsLogin(true)} />
  );
}