import { useMemo, useRef, useState, useEffect } from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  useNavigate,
  useParams,
} from "react-router-dom";
import { api, saveAuth, clearAuth, getToken } from "./api";
import "./App.css";

function LoginPage({ onLogin }) {
  const [id, setId] = useState("");
  const [pw, setPw] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleLogin = async () => {
    setError("");
    try {
      const res = await api.login({ username: id, password: pw });
      saveAuth(res.token, res.user);
      onLogin(res.user);
      navigate("/videos");
    } catch (e) {
      setError(e.message);
    }
  };

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
              placeholder="아이디"
              value={id}
              onChange={(e) => setId(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleLogin()}
            />
          </div>

          <div className="input-group">
            <input
              type="password"
              placeholder="비밀번호"
              value={pw}
              onChange={(e) => setPw(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleLogin()}
            />
          </div>

          {error && <p className="login-error-text">{error}</p>}

          <div className="login-options">
            <label className="remember-me">
              <input type="checkbox" /> Remember me
            </label>
            <a href="#none" className="support-link">Support</a>
          </div>

          <button className="login-submit-btn" onClick={handleLogin}>로그인</button>
          <button
            className="login-signup-btn"
            onClick={() => navigate("/signup")}
          >
            회원가입
          </button>
        </div>
      </div>
    </div>
  );
}

function SignupPage() {
  const [form, setForm] = useState({
    username: "",
    name: "",
    email: "",
    password: "",
    passwordConfirm: "",
  });
  const [error, setError] = useState("");
  const [done, setDone] = useState(false);
  const navigate = useNavigate();

  const update = (key) => (e) =>
    setForm((prev) => ({ ...prev, [key]: e.target.value }));

  const handleSignup = async () => {
    setError("");
    if (!form.username || !form.name || !form.password) {
      setError("아이디, 이름, 비밀번호는 필수입니다.");
      return;
    }
    if (form.password !== form.passwordConfirm) {
      setError("비밀번호가 일치하지 않습니다.");
      return;
    }
    try {
      await api.signup({
        username: form.username,
        name: form.name,
        email: form.email || null,
        password: form.password,
      });
      setDone(true);
      setTimeout(() => navigate("/login"), 1200);
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <div className="login-layout">
      <div className="login-left-panel">
        <div className="brand-content">
          <div className="brand-badge">Parking Scratch Detection</div>
          <h1>회원가입</h1>
          <p>관리자 계정을 생성하여 시스템에 접속하세요.</p>
        </div>
      </div>

      <div className="login-right-panel">
        <div className="login-form-wrapper">
          <h2>회원가입</h2>
          <p className="login-subtitle">새 계정을 만듭니다.</p>

          <div className="input-group">
            <input type="text" placeholder="아이디 *" value={form.username} onChange={update("username")} />
          </div>
          <div className="input-group">
            <input type="text" placeholder="이름 *" value={form.name} onChange={update("name")} />
          </div>
          <div className="input-group">
            <input type="email" placeholder="이메일" value={form.email} onChange={update("email")} />
          </div>
          <div className="input-group">
            <input type="password" placeholder="비밀번호 *" value={form.password} onChange={update("password")} />
          </div>
          <div className="input-group">
            <input type="password" placeholder="비밀번호 확인 *" value={form.passwordConfirm} onChange={update("passwordConfirm")} />
          </div>

          {error && <p className="login-error-text">{error}</p>}
          {done && <p className="login-success-text">가입 완료! 로그인 화면으로 이동합니다.</p>}

          <button className="login-submit-btn" onClick={handleSignup}>가입하기</button>
          <button className="login-signup-btn" onClick={() => navigate("/login")}>
            로그인으로 돌아가기
          </button>
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

function formatActualTime(dateText, startTime, seconds) {
  const [hours, minutes] = startTime.split(":").map(Number);
  const date = new Date(`${dateText}T00:00:00`);
  date.setHours(hours, minutes, seconds, 0);

  return date.toLocaleTimeString("ko-KR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function formatMonthLabel(date) {
  return `${date.getFullYear()}년 ${date.getMonth() + 1}월`;
}

function getCalendarDays(monthDate) {
  const year = monthDate.getFullYear();
  const month = monthDate.getMonth();
  const firstDay = new Date(year, month, 1);
  const lastDay = new Date(year, month + 1, 0);
  const days = [];

  for (let index = 0; index < firstDay.getDay(); index += 1) {
    days.push(null);
  }

  for (let day = 1; day <= lastDay.getDate(); day += 1) {
    days.push(new Date(year, month, day));
  }

  return days;
}

function getVideosByDateApi(videos) {
  return videos.reduce((groups, video) => {
    groups[video.date] = [...(groups[video.date] ?? []), video];
    return groups;
  }, {});
}

// 2. 이벤트 통계 분석 뷰 (Custom SVG Charts)
function AnalyticsView({ filteredVideos, filterDays, setFilterDays }) {
  const stats = useMemo(() => {
    let totalEvents = 0;
    let pendingCount = 0;
    let resolvedCount = 0;

    const dateCounts = {};
    const cameraCounts = {};
    const eventTypeCounts = {};
    const hourCounts = Array(24).fill(0);

    // filteredVideos는 이미 API에서 정규화된 영상 목록
    const matchingVideos = filteredVideos;

    matchingVideos.forEach(video => {
      const [startHour] = video.startTime.split(":").map(Number);

      video.events.forEach(event => {
        totalEvents += 1;

        // Status counts
        if (event.status === "확인 필요") pendingCount++;
        else if (event.status === "분석 완료") resolvedCount++;

        // Date grouping
        dateCounts[video.date] = (dateCounts[video.date] || 0) + 1;

        // Camera grouping
        cameraCounts[video.camera] = (cameraCounts[video.camera] || 0) + 1;

        // Type grouping
        let type = "기타";
        if (event.title.includes("스크래치") || event.title.includes("긁힘")) type = "스크래치 의심";
        else if (event.title.includes("문콕")) type = "문콕 접촉 의심";
        else if (event.title.includes("충돌") || event.title.includes("접촉")) type = "차량 충돌 의심";
        else if (event.title.includes("접근") || event.title.includes("감지")) type = "인물 접근 감지";

        eventTypeCounts[type] = (eventTypeCounts[type] || 0) + 1;

        // Hour computation
        const eventSeconds = event.timestamp;
        const eventHour = (startHour + Math.floor(eventSeconds / 3600)) % 24;
        hourCounts[eventHour]++;
      });
    });

    // Sort dates chronologically
    const sortedDates = Object.keys(dateCounts).sort().map(date => ({
      date: date.substring(5), // MM-DD
      count: dateCounts[date]
    }));

    // Convert cameras to array
    const cameraList = Object.keys(cameraCounts).map(cam => ({
      name: cam,
      count: cameraCounts[cam]
    })).sort((a, b) => b.count - a.count);

    // Convert event types to array
    const eventTypeList = Object.keys(eventTypeCounts).map(type => ({
      name: type,
      count: eventTypeCounts[type]
    }));

    // Hourly grouped brackets
    const hourlyGroups = [
      { name: "새벽 (00-06)", count: 0 },
      { name: "오전 (06-12)", count: 0 },
      { name: "오후 (12-18)", count: 0 },
      { name: "야간 (18-24)", count: 0 }
    ];
    for (let h = 0; h < 24; h++) {
      const count = hourCounts[h];
      if (h < 6) hourlyGroups[0].count += count;
      else if (h < 12) hourlyGroups[1].count += count;
      else if (h < 18) hourlyGroups[2].count += count;
      else hourlyGroups[3].count += count;
    }

    return {
      totalVideos: matchingVideos.length,
      totalEvents,
      pendingCount,
      resolvedCount,
      sortedDates,
      cameraList,
      eventTypeList,
      hourlyGroups
    };
  }, [filteredVideos]);

  const dailyTrendChart = useMemo(() => {
    const width = 500;
    const height = 200;
    const paddingX = 40;
    const paddingY = 30;
    const plotW = width - paddingX * 2;
    const plotH = height - paddingY * 2;
    const data = stats.sortedDates;
    if (data.length === 0) return null;

    const maxVal = Math.max(...data.map(d => d.count), 4) + 1;

    const points = data.map((d, i) => {
      const x = paddingX + (i * plotW) / (data.length - 1);
      const y = height - paddingY - (d.count * plotH) / maxVal;
      return { x, y, label: d.date, value: d.count };
    });

    const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
    const areaPath = `${linePath} L ${points[points.length - 1].x} ${height - paddingY} L ${points[0].x} ${height - paddingY} Z`;

    return (
      <svg width="100%" height="200" viewBox={`0 0 ${width} ${height}`} className="stats-svg">
        <defs>
          <linearGradient id="area-gradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--chart-primary)" stopOpacity="0.4" />
            <stop offset="100%" stopColor="var(--chart-primary)" stopOpacity="0.0" />
          </linearGradient>
        </defs>

        {/* Y Axis Grid lines */}
        {[0, 1, 2, 3, 4, 5].map(v => {
          const y = height - paddingY - (v * plotH) / maxVal;
          return (
            <g key={v}>
              <line x1={paddingX} y1={y} x2={width - paddingX} y2={y} stroke="var(--chart-grid)" strokeDasharray="3 3" />
              <text x={paddingX - 10} y={y + 4} textAnchor="end" className="chart-axis-text" fill="var(--chart-text)">{v}</text>
            </g>
          );
        })}

        {/* Area */}
        <path d={areaPath} fill="url(#area-gradient)" />

        {/* Line */}
        <path d={linePath} fill="none" stroke="var(--chart-primary)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />

        {/* Dots */}
        {points.map((p, i) => (
          <g key={i}>
            <circle cx={p.x} cy={p.y} r="4" fill="var(--chart-primary)" stroke="var(--chart-grid)" strokeWidth="1" />
            <text x={p.x} y={p.y - 10} textAnchor="middle" className="chart-value-text" fill="var(--chart-text-primary)">
              {p.value}
            </text>
            <text x={p.x} y={height - paddingY + 16} textAnchor="middle" className="chart-axis-text" fill="var(--chart-text)">
              {p.label}
            </text>
          </g>
        ))}
      </svg>
    );
  }, [stats.sortedDates]);

  const donutChart = useMemo(() => {
    const total = stats.totalEvents;
    if (total === 0) return null;
    const r = 50;
    const circ = 2 * Math.PI * r;
    let currentAngle = -90;

    const colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

    const segments = [];
    for (let idx = 0; idx < stats.eventTypeList.length; idx++) {
      const type = stats.eventTypeList[idx];
      const percentage = (type.count / total) * 100;
      const angle = (type.count / total) * 360;
      const strokeDashoffset = circ - (type.count / total) * circ;
      const rotation = currentAngle;
      currentAngle += angle;
      const color = colors[idx % colors.length];

      segments.push({
        ...type,
        percentage: percentage.toFixed(1),
        strokeDashoffset,
        rotation,
        color
      });
    }

    return (
      <div className="donut-chart-container">
        <svg width="180" height="180" viewBox="0 0 200 200">
          <circle cx="100" cy="100" r={r} fill="transparent" stroke="var(--chart-grid)" strokeWidth="16" />
          {segments.map((seg, idx) => (
            <circle
              key={idx}
              cx="100"
              cy="100"
              r={r}
              fill="transparent"
              stroke={seg.color}
              strokeWidth="18"
              strokeDasharray={circ}
              strokeDashoffset={seg.strokeDashoffset}
              transform={`rotate(${seg.rotation} 100 100)`}
              className="donut-segment"
            />
          ))}
          <g className="donut-center-text">
            <text x="100" y="95" textAnchor="middle" className="donut-total" fill="var(--chart-text-primary)" style={{ fontSize: "28px", fontWeight: "800" }}>
              {total}
            </text>
            <text x="100" y="115" textAnchor="middle" className="donut-label" fill="var(--chart-text)" style={{ fontSize: "12px", fontWeight: "600" }}>
              총 감지 건수
            </text>
          </g>
        </svg>

        <div className="donut-legend">
          {segments.map((seg, idx) => (
            <div key={idx} className="legend-item">
              <span className="legend-badge" style={{ backgroundColor: seg.color }} />
              <span className="legend-name">{seg.name}</span>
              <span className="legend-count">{seg.count}건 ({seg.percentage}%)</span>
            </div>
          ))}
        </div>
      </div>
    );
  }, [stats.eventTypeList, stats.totalEvents]);

  const hourlyChartSvg = useMemo(() => {
    const width = 450;
    const height = 200;
    const paddingX = 40;
    const paddingY = 30;
    const plotW = width - paddingX * 2;
    const plotH = height - paddingY * 2;
    const data = stats.hourlyGroups;
    const maxVal = Math.max(...data.map(d => d.count), 4) + 1;

    const barW = 32;
    const gap = (plotW - barW * data.length) / (data.length - 1);

    return (
      <svg width="100%" height="200" viewBox={`0 0 ${width} ${height}`} className="stats-svg">
        {[0, 1, 2, 3, 4, 5].map(v => {
          const y = height - paddingY - (v * plotH) / maxVal;
          return (
            <g key={v}>
              <line x1={paddingX} y1={y} x2={width - paddingX} y2={y} stroke="var(--chart-grid)" strokeDasharray="3 3" />
              <text x={paddingX - 10} y={y + 4} textAnchor="end" className="chart-axis-text" fill="var(--chart-text)">{v}</text>
            </g>
          );
        })}

        {data.map((d, i) => {
          const barH = (d.count * plotH) / maxVal;
          const x = paddingX + i * (barW + gap);
          const y = height - paddingY - barH;

          return (
            <g key={i}>
              <rect
                x={x}
                y={y}
                width={barW}
                height={barH}
                fill="var(--chart-secondary)"
                rx="4"
                className="chart-bar"
              />
              <text x={x + barW / 2} y={y - 6} textAnchor="middle" className="chart-value-text" fill="var(--chart-text-primary)">
                {d.count}
              </text>
              <text x={x + barW / 2} y={height - paddingY + 16} textAnchor="middle" className="chart-axis-text" fill="var(--chart-text)">
                {d.name.split(" ")[0]}
              </text>
            </g>
          );
        })}
      </svg>
    );
  }, [stats.hourlyGroups]);

  const cameraStats = useMemo(() => {
    const maxCount = Math.max(...stats.cameraList.map(c => c.count), 1);

    return (
      <div className="camera-stats-list">
        {stats.cameraList.map((cam, idx) => {
          const pct = (cam.count / maxCount) * 100;
          return (
            <div key={idx} className="camera-stat-row">
              <div className="camera-name">{cam.name}</div>
              <div className="camera-bar-wrapper">
                <div className="camera-bar-fill" style={{ width: `${pct}%` }} />
              </div>
              <div className="camera-count">{cam.count}건</div>
            </div>
          );
        })}
      </div>
    );
  }, [stats.cameraList]);

  return (
    <div className="analytics-view">
      <div className="analytics-header">
        <h2>📊 AI 감지 이벤트 통계 분석</h2>
        <p>CCTV 녹화본에서 감지된 차량 사고 및 접촉 의심 이벤트 통계 요약입니다.</p>
      </div>

      {/* 기간 필터 */}
      <div className="filter-pills" style={{ marginBottom: "24px" }}>
        <button className={filterDays === 7 ? "active" : ""} onClick={() => setFilterDays(7)}>1주일</button>
        <button className={filterDays === 14 ? "active" : ""} onClick={() => setFilterDays(14)}>2주일</button>
        <button className={filterDays === 30 ? "active" : ""} onClick={() => setFilterDays(30)}>1개월</button>
        <button className={filterDays === 90 ? "active" : ""} onClick={() => setFilterDays(90)}>3개월</button>
        <button className={filterDays === 9999 ? "active" : ""} onClick={() => setFilterDays(9999)}>전체</button>
      </div>

      {stats.totalEvents === 0 ? (
        <div className="empty-state" style={{ padding: "80px 20px" }}>선택한 기간에 감지된 이벤트 통계 데이터가 없습니다.</div>
      ) : (
        <>
          <div className="analytics-summary-cards">
            <div className="summary-card">
              <div className="card-icon">🎥</div>
              <div className="card-data">
                <span className="card-label">분석된 총 영상</span>
                <span className="card-value">{stats.totalVideos}개</span>
              </div>
            </div>
            <div className="summary-card">
              <div className="card-icon">🚨</div>
              <div className="card-data">
                <span className="card-label">누적 감지 이벤트</span>
                <span className="card-value">{stats.totalEvents}건</span>
              </div>
            </div>
            <div className="summary-card warning">
              <div className="card-icon">⚠️</div>
              <div className="card-data">
                <span className="card-label">확인 필요 이벤트</span>
                <span className="card-value">{stats.pendingCount}건</span>
              </div>
            </div>
          </div>

          <div className="analytics-grid">
            <div className="analytics-chart-card">
              <h3> 날짜별 감지 이벤트 추이</h3>
              <div className="chart-wrapper">{dailyTrendChart}</div>
            </div>

            <div className="analytics-chart-card">
              <h3> 이벤트 유형별 비율</h3>
              <div className="chart-wrapper donut-wrapper">{donutChart}</div>
            </div>

            <div className="analytics-chart-card">
              <h3> 시간대별 발생 빈도</h3>
              <div className="chart-wrapper">{hourlyChartSvg}</div>
            </div>

            <div className="analytics-chart-card">
              <h3> 카메라별 감지 현황</h3>
              <div className="chart-wrapper">{cameraStats}</div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// 대시보드 컴포넌트
function Dashboard({ onLogout, view }) {
  const navigate = useNavigate();
  const { videoId } = useParams();
  const currentView = view === "analytics" ? "analytics" : "dashboard";

  // 상태 관리
  const [videos, setVideos] = useState([]); // API에서 로드한 영상 목록
  const [filterDays, setFilterDays] = useState(7); // 기본 1주일
  const [selectedVideo, setSelectedVideo] = useState(null); // null이면 홈(그리드) 화면, 값이 있으면 영상 재생 화면
  const [currentEventId, setCurrentEventId] = useState(null); // 현재 선택된 이벤트 마커
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0); // 실제 영상 재생 위치(초)
  const [selectedCalendarDate, setSelectedCalendarDate] = useState(null); // 달력에서 선택한 날짜
  const [playbackSpeed, setPlaybackSpeed] = useState("1");
  const [volume, setVolume] = useState(70);
  const [quality, setQuality] = useState("auto");
  const [isTheaterMode, setIsTheaterMode] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [calendarMonth, setCalendarMonth] = useState(new Date("2026-05-01T00:00:00"));
  const playerContainerRef = useRef(null);

  // 바운딩박스 관련 상태
  const [isBBoxMode, setIsBBoxMode] = useState(false);
  const [bboxList, setBboxList] = useState([]);
  const [currentDraw, setCurrentDraw] = useState(null); // { startX, startY, endX, endY }
  const [isDrawing, setIsDrawing] = useState(false);
  const bboxOverlayRef = useRef(null);
  const videoElRef = useRef(null); // 실제 <video> 엘리먼트 (원본 해상도 환산용)
  const [showBBoxPanel, setShowBBoxPanel] = useState(false);

  // 사고감지 실행 / 분석 관련 상태
  const [showDetectConfirm, setShowDetectConfirm] = useState(false); // 확인 팝오버
  const [toast, setToast] = useState(null); // { type, message }
  // 동시 분석 작업 목록: [{ taskId, videoId, dateLabel, cameraLabel, estimatedSec, startedAt }]
  const [analyzingJobs, setAnalyzingJobs] = useState([]);
  const [showJobsDropdown, setShowJobsDropdown] = useState(false);
  const [now, setNow] = useState(() => Date.now()); // 진행률 계산용 현재시각(0.5s마다 갱신)
  const selectedVideoIdRef = useRef(null); // 폴링 중 최신 선택 영상 추적(stale closure 방지)
  const [clipEvent, setClipEvent] = useState(null); // CAM 클립 팝업 대상 이벤트

  // 업로드 모달
  const [showUpload, setShowUpload] = useState(false);

  const showToast = (message, type = "warning") => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 2600);
  };

  // 프로필 & 설정 관련 상태
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [activeSettingsTab, setActiveSettingsTab] = useState("account"); // "account", "security"

  // 설정 정보
  const adminName = "admin";
  const [adminRealName, setAdminRealName] = useState("홍길동");
  const [adminPhone, setAdminPhone] = useState("010-1234-5678");
  const [adminEmail, setAdminEmail] = useState("admin@cbnu-capstone.com");

  // 비밀번호 변경 필드
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  // 테마 설정 (다크 모드)
  const [isDarkMode, setIsDarkMode] = useState(false);

  // 다크 모드 활성화 / 비활성화 제어
  useEffect(() => {
    if (isDarkMode) {
      document.body.classList.add("dark-mode");
    } else {
      document.body.classList.remove("dark-mode");
    }
  }, [isDarkMode]);

  // API에서 영상 목록 로드
  const loadVideos = async () => {
    try {
      const list = await api.listVideos(9999);
      setVideos(list);
      return list;
    } catch (e) {
      showToast(`영상 목록을 불러오지 못했습니다: ${e.message}`, "error");
      return [];
    }
  };

  useEffect(() => {
    // 마운트 시 영상 목록 비동기 로드 (setState는 await 이후 발생)
    // eslint-disable-next-line react-hooks/exhaustive-deps, react-hooks/set-state-in-effect
    loadVideos();
  }, []);

  // 라우트(videoId)에 맞춰 selectedVideo 설정
  useEffect(() => {
    if (view !== "watch" || !videoId) {
      // watch 화면이 아니면 선택 영상 해제 (라우트 동기화 가드)
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setSelectedVideo(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const v = await api.getVideo(videoId);
        if (cancelled) return;
        setSelectedVideo(v);
        setCurrentEventId(null);
        setIsPlaying(false);
        setIsTheaterMode(false);
        setIsSidebarOpen(true);
        setBboxList([]);
        setIsBBoxMode(false);
        setShowDetectConfirm(false);
        setCurrentTime(0);
        if (v.date) {
          setCalendarMonth(new Date(`${v.date}T00:00:00`));
          setSelectedCalendarDate(v.date);
        }
      } catch (e) {
        showToast(`영상을 불러오지 못했습니다: ${e.message}`, "error");
        navigate("/videos");
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, videoId]);

  // 날짜 필터링 계산
  const filteredVideos = useMemo(() => {
    const today = new Date();
    const startDate = formatDate(addDays(today, -filterDays));
    if (filterDays >= 9999) return videos;
    return videos.filter((video) => video.date >= startDate);
  }, [filterDays, videos]);

  const videosByDate = useMemo(() => {
    return getVideosByDateApi(videos);
  }, [videos]);

  const calendarDays = useMemo(() => getCalendarDays(calendarMonth), [calendarMonth]);
  const nextVideos = useMemo(() => {
    if (!selectedVideo || videos.length === 0) return [];
    const currentIndex = videos.findIndex((video) => video.id === selectedVideo.id);
    if (currentIndex === -1) return [];
    return Array.from({ length: Math.min(4, videos.length - 1) }, (_, index) => {
      return videos[(currentIndex + index + 1) % videos.length];
    });
  }, [selectedVideo, videos]);

  const moveCalendarMonth = (offset) => {
    setCalendarMonth((month) => new Date(month.getFullYear(), month.getMonth() + offset, 1));
  };

  const handleCalendarDateClick = (date) => {
    const dateText = formatDate(date);
    setSelectedCalendarDate(dateText); // 영상 유무와 무관하게 날짜 선택 표시
    const dayVideos = videosByDate[dateText] ?? [];
    if (dayVideos.length > 0) {
      handleWatchVideo(dayVideos[0]);
    }
  };

  // 영상을 클릭하여 시청 모드로 진입 → URL 이동
  const handleWatchVideo = (video) => {
    navigate(`/videos/${video.id}`);
  };

  // 홈(목록)으로 돌아가기
  const handleBackToHome = () => {
    navigate("/videos");
  };

  // '사고감지 실행' 클릭 → bbox 없으면 경고, 있으면 확인 팝오버
  const handleDetectClick = () => {
    if (bboxList.length === 0) {
      showToast("사고를 감지할 차량을 드래그하여 주십시오.", "warning");
      return;
    }
    setShowDetectConfirm(true);
  };

  // 분석 태스크 상태 폴링 (analyzedId = 분석을 시작한 영상 id)
  const removeJob = (taskId) =>
    setAnalyzingJobs((prev) => prev.filter((j) => j.taskId !== taskId));

  const pollTask = (taskId, analyzedId) => {
    const poll = async () => {
      try {
        const status = await api.taskStatus(taskId);
        if (status.status === "SUCCESS") {
          removeJob(taskId);
          showToast(`분석 완료: 사고 의심 구간 ${status.events.length}건`, "success");
          // 분석한 영상을 아직 보고 있을 때만 화면 갱신 (다른 영상으로 이동했으면 덮어쓰지 않음)
          if (selectedVideoIdRef.current === analyzedId) {
            const v = await api.getVideo(analyzedId);
            setSelectedVideo(v);
          }
          loadVideos();
          return;
        }
        if (status.status === "FAILURE") {
          removeJob(taskId);
          showToast(`분석 실패: ${status.error_message || "오류"}`, "error");
          return;
        }
        setTimeout(poll, 2000); // PENDING/PROCESSING → 재시도
      } catch (e) {
        removeJob(taskId);
        showToast(`상태 조회 실패: ${e.message}`, "error");
      }
    };
    poll();
  };

  // 확인 팝오버에서 '실행' → bbox 원본해상도 환산 후 분석 요청
  const runDetection = async () => {
    setShowDetectConfirm(false);
    const box = bboxList[0];
    if (!box) return;

    // 화면(오버레이) 좌표 → 원본 영상 해상도 픽셀 좌표 환산
    const videoEl = videoElRef.current;
    let scaleX = 1;
    let scaleY = 1;
    if (videoEl && videoEl.videoWidth && videoEl.clientWidth) {
      scaleX = videoEl.videoWidth / videoEl.clientWidth;
      scaleY = videoEl.videoHeight / videoEl.clientHeight;
    }
    const bbox = {
      bbox_xmin: Math.round(box.xmin * scaleX),
      bbox_ymin: Math.round(box.ymin * scaleY),
      bbox_xmax: Math.round(box.xmax * scaleX),
      bbox_ymax: Math.round(box.ymax * scaleY),
    };

    // 분석 시작: 지정 모드를 끄고 해당 영상을 '분석 중'으로 표시
    const analyzedId = selectedVideo.id;
    setIsBBoxMode(false);
    setCurrentDraw(null);
    setIsDrawing(false);

    // 예상 소요시간(초) 추정: 슬라이딩 윈도우 수(총프레임/stride) 기반 (CPU ~1s/윈도우)
    const totalFrames = Math.round(
      (selectedVideo.duration || 0) * (selectedVideo.fps || 30));
    const estimatedSec = Math.max(10, Math.round((totalFrames / 15) * 1.1));
    const job = {
      taskId: null,
      videoId: analyzedId,
      dateLabel: selectedVideo.date,
      cameraLabel: selectedVideo.camera,
      estimatedSec,
      startedAt: Date.now(),
    };

    try {
      const res = await api.analyze(analyzedId, bbox);
      job.taskId = res.task_id;
      setAnalyzingJobs((prev) => [...prev, job]);
      pollTask(res.task_id, analyzedId);
    } catch (e) {
      showToast(`분석 요청 실패: ${e.message}`, "error");
    }
  };

  const handleToggleFullscreen = () => {
    const player = playerContainerRef.current;
    if (!player) return;

    if (document.fullscreenElement) {
      document.exitFullscreen();
      return;
    }

    player.requestFullscreen?.();
  };

  // ── 실제 <video> 재생 제어 ──────────────────────────────
  const togglePlay = () => {
    const v = videoElRef.current;
    if (!v) return;
    if (v.paused) v.play();
    else v.pause();
  };

  // 재생바 클릭으로 탐색(seek)
  const handleSeek = (e) => {
    const v = videoElRef.current;
    if (!v || !v.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    v.currentTime = ratio * v.duration;
  };

  // 이벤트 마커 클릭 → 해당 시점으로 이동 후 재생
  const seekToEvent = (event) => {
    setCurrentEventId(event.id);
    const v = videoElRef.current;
    if (v) {
      v.currentTime = event.timestamp;
      v.play();
    }
  };

  // 배속/볼륨을 실제 video에 반영
  useEffect(() => {
    const v = videoElRef.current;
    if (v) v.playbackRate = parseFloat(playbackSpeed);
  }, [playbackSpeed, selectedVideo]);

  useEffect(() => {
    const v = videoElRef.current;
    if (v) v.volume = volume / 100;
  }, [volume, selectedVideo]);

  // 폴링 콜백에서 현재 보고 있는 영상 id를 최신으로 참조 (stale closure 방지)
  useEffect(() => {
    selectedVideoIdRef.current = selectedVideo?.id ?? null;
  }, [selectedVideo]);

  // 현재 보고 있는 영상이 분석 중인지 (영상별로 판정 — 다른 영상엔 영향 없음)
  const isAnalyzing =
    !!selectedVideo && analyzingJobs.some((j) => j.videoId === selectedVideo.id);

  // 분석 작업이 있는 동안 진행률 바를 0.5초마다 갱신
  useEffect(() => {
    if (analyzingJobs.length === 0) return;
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, [analyzingJobs.length]);

  // 작업별 진행률(%) / 남은 시간(초) 계산 (시간 기반 추정 — 완료 전까지 96%에서 대기)
  const jobProgress = (job) => {
    const elapsed = (now - job.startedAt) / 1000;
    const pct = Math.min(96, (elapsed / job.estimatedSec) * 100);
    const remain = Math.max(0, Math.ceil(job.estimatedSec - elapsed));
    return { pct, remain };
  };

  return (
    <div className="dashboard-layout">
      {/* 상단 네비게이션 바 */}
      <header className="top-navbar">
        <div className="nav-section nav-left">
          <button className="nav-logo nav-home-btn" onClick={handleBackToHome}>SIOT</button>
        </div>
        <div className="nav-section nav-center">
          {analyzingJobs.length === 0 ? (
            <div className="analysis-status-box analysis-status-idle">
              <span className="analysis-idle-text">현재 분석 중인 영상이 없습니다</span>
            </div>
          ) : (() => {
            const lead = analyzingJobs[0];
            const { pct, remain } = jobProgress(lead);
            return (
              <div className="analysis-status-box">
                <div className="analysis-status-main">
                  <div className="analysis-status-info">
                    <span className="analysis-status-label">
                      🔍 {lead.dateLabel} 분석 중
                    </span>
                    <span className="analysis-status-eta">약 {remain}초 남음</span>
                  </div>
                  <div className="analysis-progress-track">
                    <div
                      className="analysis-progress-fill"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>

                <button
                  className="analysis-jobs-toggle"
                  onClick={() => setShowJobsDropdown((v) => !v)}
                  title="분석 중인 영상 목록"
                >
                  {analyzingJobs.length}개 ▾
                </button>

                {showJobsDropdown && (
                  <>
                    <div
                      className="dropdown-overlay"
                      onClick={() => setShowJobsDropdown(false)}
                    />
                    <div className="analysis-jobs-dropdown">
                      <div className="analysis-jobs-dropdown-title">
                        분석 중인 영상 ({analyzingJobs.length})
                      </div>
                      {analyzingJobs.map((job) => {
                        const p = jobProgress(job);
                        return (
                          <div className="analysis-job-row" key={job.taskId ?? job.startedAt}>
                            <div className="analysis-job-top">
                              <span className="analysis-job-name">
                                {job.dateLabel} · {job.cameraLabel}
                              </span>
                              <span className="analysis-job-pct">{Math.round(p.pct)}%</span>
                            </div>
                            <div className="analysis-progress-track">
                              <div
                                className="analysis-progress-fill"
                                style={{ width: `${p.pct}%` }}
                              />
                            </div>
                            <span className="analysis-job-eta">약 {p.remain}초 남음</span>
                          </div>
                        );
                      })}
                    </div>
                  </>
                )}
              </div>
            );
          })()}
        </div>
        <div className="nav-section nav-right">
          <div className="profile-menu-container">
            <button
              className="profile-trigger-btn simple-avatar-trigger"
              onClick={() => setIsProfileOpen(!isProfileOpen)}
              aria-label="프로필 메뉴 열기"
            >
              <div className="profile-avatar">
                <span>A</span>
              </div>
            </button>

            {isProfileOpen && (
              <>
                <div className="dropdown-overlay" onClick={() => setIsProfileOpen(false)} />
                <div className="profile-dropdown-menu">
                  {/* 관리자 정보 요약 Header */}
                  <div className="dropdown-header">
                    <div className="header-avatar">A</div>
                    <div className="header-info">
                      <span className="info-name">{adminRealName} ({adminName})</span>
                      <span className="info-role">시스템 관리자</span>
                    </div>
                  </div>

                  <div className="dropdown-divider" />

                  {/* 내 정보 설정 (Account Settings) */}
                  <div className="dropdown-section-title">내 정보 설정</div>
                  <button
                    className="dropdown-item"
                    onClick={() => {
                      setIsProfileOpen(false);
                      setIsSettingsOpen(true);
                      setActiveSettingsTab("account");
                    }}
                  >
                    👤 관리자 정보 수정
                  </button>
                  <button
                    className="dropdown-item"
                    onClick={() => {
                      setIsProfileOpen(false);
                      setIsSettingsOpen(true);
                      setActiveSettingsTab("security");
                    }}
                  >
                    🔑 비밀번호 변경
                  </button>

                  <div className="dropdown-divider" />

                  {/* 화면 이동 */}
                  <div className="dropdown-section-title">화면 이동</div>
                  <button
                    className={`dropdown-item ${currentView === "analytics" ? "active-menu-item" : ""}`}
                    onClick={() => {
                      setIsProfileOpen(false);
                      navigate("/analytics");
                    }}
                  >
                    📊 이벤트 통계 그래프
                  </button>

                  <div className="dropdown-divider" />

                  {/* 테마 설정 (Appearance) */}
                  <div className="dropdown-section-title">테마 설정</div>
                  <div className="dropdown-item-toggle">
                    <span>🌙 다크 모드</span>
                    <label className="switch-mini">
                      <input
                        type="checkbox"
                        checked={isDarkMode}
                        onChange={(e) => setIsDarkMode(e.target.checked)}
                      />
                      <span className="slider-mini round"></span>
                    </label>
                  </div>

                  <div className="dropdown-divider" />

                  {/* 로그아웃 (Logout) */}
                  <button
                    className="dropdown-item logout-item"
                    onClick={() => {
                      setIsProfileOpen(false);
                      onLogout();
                    }}
                  >
                    🚪 로그아웃
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </header>

      {/* 컨텐츠 영역: 통계 뷰 또는 모니터링 뷰 */}
      {currentView === "analytics" ? (
        <AnalyticsView
          filteredVideos={filteredVideos}
          filterDays={filterDays}
          setFilterDays={setFilterDays}
        />
      ) : !selectedVideo ? (
        <main className="home-view">
          {/* 기간 필터 + 업로드 버튼 */}
          <div className="home-toolbar">
            <div className="filter-pills">
              <button className={filterDays === 7 ? "active" : ""} onClick={() => setFilterDays(7)}>1주일</button>
              <button className={filterDays === 14 ? "active" : ""} onClick={() => setFilterDays(14)}>2주일</button>
              <button className={filterDays === 30 ? "active" : ""} onClick={() => setFilterDays(30)}>1개월</button>
              <button className={filterDays === 90 ? "active" : ""} onClick={() => setFilterDays(90)}>3개월</button>
              <button className={filterDays === 9999 ? "active" : ""} onClick={() => setFilterDays(9999)}>전체</button>
            </div>
            <button className="upload-open-btn" onClick={() => setShowUpload(true)}>
              ⬆ 영상 업로드
            </button>
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
        <main className={`watch-view ${isTheaterMode ? "theater-view" : ""}`}>
          <div className="watch-header">
            <button className="back-btn" onClick={handleBackToHome}>
              ← 목록으로 돌아가기
            </button>

            {/* 영상(player) 컬럼에 맞춰 정렬되는 오른쪽 영역 */}
            <div className="watch-header-right">
            {/* 사고 차량 지정 + 사고감지 실행 버튼 그룹 (나란히) */}
            <div className="watch-header-actions">
            <button
              className={`bbox-designate-btn ${isBBoxMode ? "active" : ""}`}
              disabled={isAnalyzing}
              onClick={() => {
                setIsBBoxMode((prev) => {
                  if (prev) {
                    // 모드 해제 시 진행 중인 드래그 초기화
                    setCurrentDraw(null);
                    setIsDrawing(false);
                  } else {
                    setShowBBoxPanel(true);
                  }
                  return !prev;
                });
              }}
              title={isAnalyzing ? "분석 중에는 지정할 수 없습니다" : (isBBoxMode ? "바운딩박스 모드 종료" : "사고 차량 바운딩박스 지정 모드 시작")}
            >
              {isBBoxMode ? "지정 모드 종료" : "사고 차량 지정하기"}
            </button>

            {/* 사고감지 실행 버튼 + 확인 팝오버 */}
            <div className="detect-btn-wrapper">
              <button
                className="detect-run-btn"
                onClick={handleDetectClick}
                disabled={isAnalyzing}
                title="선택한 차량의 사고예상 구간 탐지"
              >
                {isAnalyzing ? "분석 중..." : "사고감지 실행"}
              </button>

              {showDetectConfirm && (
                <>
                  <div
                    className="detect-popover-overlay"
                    onClick={() => setShowDetectConfirm(false)}
                  />
                  <div className="detect-popover">
                    <p className="detect-popover-text">
                      현재 선택하신 차량의 사고예상 구간을 탐지하시겠습니까?
                    </p>
                    <div className="detect-popover-actions">
                      <button className="detect-popover-run" onClick={runDetection}>
                        실행
                      </button>
                      <button
                        className="detect-popover-cancel"
                        onClick={() => setShowDetectConfirm(false)}
                      >
                        뒤로가기
                      </button>
                    </div>
                  </div>
                </>
              )}
            </div>
            </div>

            <div className="watch-header-metadata">
              <span className="metadata-item"><strong>카메라:</strong> {selectedVideo.camera}</span>
              <span className="metadata-item"><strong>녹화 일시:</strong> {selectedVideo.date} {selectedVideo.startTime}</span>
              <span className="metadata-item"><strong>총 이벤트:</strong> {selectedVideo.events.length}건 감지됨</span>
            </div>
            </div>
          </div>

          <div className={`watch-layout ${isTheaterMode ? "theater-mode" : ""} ${!isSidebarOpen ? "sidebar-collapsed" : ""}`}>
            <aside className={`event-sidebar ${!isSidebarOpen ? "collapsed" : ""}`}>
              {isSidebarOpen ? (
                <>
                  <div className="event-sidebar-header">
                    <div>
                      <h3>감지된 이벤트 목록</h3>
                      <p className="event-summary">총 {selectedVideo.events.length}건의 이벤트가 있습니다.</p>
                    </div>
                    <button
                      className="sidebar-icon-btn"
                      onClick={() => setIsSidebarOpen(false)}
                      aria-label="이벤트 목록 닫기"
                    >
                      ×
                    </button>
                  </div>

                  <div className="event-list">
                    {selectedVideo.events.map((event) => (
                      <div
                        key={event.id}
                        className={`event-item ${currentEventId === event.id ? "active" : ""}`}
                        onClick={() => setCurrentEventId(event.id)}
                      >
                        <div className="event-item-top">
                          <span className="event-time-pill">{formatTime(event.timestamp)}</span>
                          <span className="event-actual-time-text">
                            {formatActualTime(selectedVideo.date, selectedVideo.startTime, event.timestamp)}
                          </span>
                          <button
                            className="event-play-icon-btn"
                            onClick={(e) => {
                              e.stopPropagation();
                              setCurrentEventId(event.id);
                              if (event.hasClip) {
                                setClipEvent(event);
                              } else {
                                showToast("이 이벤트의 CAM 클립이 아직 없습니다.", "warning");
                              }
                            }}
                            aria-label={`${formatTime(event.timestamp)} 사고구간 CAM 클립 재생`}
                            title="사고구간 CAM 클립 보기"
                          >
                            ▶
                          </button>
                        </div>
                        <div className="event-item-bottom">
                          <h4 className="event-title-text">{event.title}</h4>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* 달력을 사이드바 이벤트 목록 하단으로 삽입 */}
                  <div style={{ marginTop: "16px", paddingTop: "12px", borderTop: "1px solid #e2e8f0" }}>
                    <section className="event-calendar-panel" style={{ background: "transparent", padding: 0 }}>
                      <div className="calendar-header">
                        <button className="calendar-nav-btn" onClick={() => moveCalendarMonth(-1)} aria-label="이전 달">
                          ‹
                        </button>
                        <h2>{formatMonthLabel(calendarMonth)}</h2>
                        <button className="calendar-nav-btn" onClick={() => moveCalendarMonth(1)} aria-label="다음 달">
                          ›
                        </button>
                      </div>

                      <div className="calendar-weekdays">
                        {["일", "월", "화", "수", "목", "금", "토"].map((day) => (
                          <span key={day}>{day}</span>
                        ))}
                      </div>

                      <div className="calendar-grid">
                        {calendarDays.map((date, index) => {
                          if (!date) {
                            return <div key={`empty-${index}`} className="calendar-day empty" />;
                          }

                          const dateText = formatDate(date);
                          const videos = videosByDate[dateText] ?? [];
                          const hasVideo = videos.length > 0;
                          const isSelectedDate = (selectedCalendarDate ?? selectedVideo.date) === dateText;
                          
                          // 영상의 총 이벤트 건수를 합산하여 강도를 계산
                          const totalEvents = videos.reduce((acc, v) => acc + v.events.length, 0);

                          let intensityClass = "";
                          if (totalEvents === 1) intensityClass = "intensity-1";
                          else if (totalEvents === 2) intensityClass = "intensity-2";
                          else if (totalEvents >= 3) intensityClass = "intensity-3";

                          return (
                            <button
                              key={dateText}
                              className={`calendar-day ${hasVideo ? `has-video ${intensityClass}` : ""} ${isSelectedDate ? "selected" : ""}`}
                              onClick={() => handleCalendarDateClick(date)}
                              title={hasVideo ? `총 ${totalEvents}건의 이벤트` : "영상 없음"}
                            >
                              <span className="calendar-date-number">{date.getDate()}</span>
                            </button>
                          );
                        })}
                      </div>
                    </section>
                  </div>
                </>
              ) : (
                <button
                  className="sidebar-open-btn"
                  onClick={() => setIsSidebarOpen(true)}
                >
                  이벤트 목록 열기
                </button>
              )}

              {!isSidebarOpen && nextVideos.length > 0 && (
                <section className="next-video-section">
                  <h3>다음 영상</h3>
                  <div className="next-video-list">
                    {nextVideos.map((video) => (
                      <button
                        key={video.id}
                        className="next-video-card"
                        onClick={() => handleWatchVideo(video)}
                      >
                        <strong>{video.date}</strong>
                        <span>{video.camera}</span>
                        <small>{video.startTime} · 이벤트 {video.events.length}건 · {formatTime(video.duration)}</small>
                      </button>
                    ))}
                  </div>
                </section>
              )}
            </aside>

            {/* 우측/중앙: 메인 비디오 플레이어 */}
            <section className="player-section">
              <div className="player-container" ref={playerContainerRef}>
                <div
                  className={`mock-video-player ${isBBoxMode ? "bbox-mode-active" : ""}`}
                  ref={bboxOverlayRef}
                  style={{ position: "relative", userSelect: "none" }}
                  onMouseDown={(e) => {
                    if (!isBBoxMode) return;
                    const rect = bboxOverlayRef.current.getBoundingClientRect();
                    const x = Math.round(e.clientX - rect.left);
                    const y = Math.round(e.clientY - rect.top);
                    setIsDrawing(true);
                    setCurrentDraw({ startX: x, startY: y, endX: x, endY: y });
                  }}
                  onMouseMove={(e) => {
                    if (!isBBoxMode || !isDrawing) return;
                    const rect = bboxOverlayRef.current.getBoundingClientRect();
                    const x = Math.round(e.clientX - rect.left);
                    const y = Math.round(e.clientY - rect.top);
                    setCurrentDraw((prev) => prev ? { ...prev, endX: x, endY: y } : null);
                  }}
                  onMouseUp={(e) => {
                    if (!isBBoxMode || !isDrawing || !currentDraw) return;
                    const rect = bboxOverlayRef.current.getBoundingClientRect();
                    const x = Math.round(e.clientX - rect.left);
                    const y = Math.round(e.clientY - rect.top);
                    const xmin = Math.min(currentDraw.startX, x);
                    const ymin = Math.min(currentDraw.startY, y);
                    const xmax = Math.max(currentDraw.startX, x);
                    const ymax = Math.max(currentDraw.startY, y);
                    if (xmax - xmin > 5 && ymax - ymin > 5) {
                      // 단일 박스만 유지 — 새로 그리면 기존 박스 교체
                      setBboxList([{ id: Date.now(), xmin, ymin, xmax, ymax }]);
                    }
                    setCurrentDraw(null);
                    setIsDrawing(false);
                  }}
                  onMouseLeave={() => {
                    if (isDrawing && currentDraw) {
                      const xmin = Math.min(currentDraw.startX, currentDraw.endX);
                      const ymin = Math.min(currentDraw.startY, currentDraw.endY);
                      const xmax = Math.max(currentDraw.startX, currentDraw.endX);
                      const ymax = Math.max(currentDraw.startY, currentDraw.endY);
                      if (xmax - xmin > 5 && ymax - ymin > 5) {
                        // 단일 박스만 유지
                        setBboxList([{ id: Date.now(), xmin, ymin, xmax, ymax }]);
                      }
                    }
                    setCurrentDraw(null);
                    setIsDrawing(false);
                  }}
                >
                  {/* 실제 업로드 영상 재생 (bbox 모드에서는 포인터 이벤트를 컨테이너로 전달) */}
                  <video
                    ref={videoElRef}
                    className="real-video"
                    src={api.streamUrl(selectedVideo.id)}
                    controls={!isBBoxMode}
                    onPlay={() => setIsPlaying(true)}
                    onPause={() => setIsPlaying(false)}
                    onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
                    style={{
                      display: "block",
                      width: "100%",
                      height: "100%",
                      objectFit: "fill",
                      background: "#000",
                      pointerEvents: isBBoxMode ? "none" : "auto",
                    }}
                  />
                  {/* 바운딩박스 모드 안내 문구 */}
                  {isBBoxMode && (
                    <div className="bbox-mode-guide">
                      <span className="bbox-guide-icon">🖱️</span>
                      <span>드래그하여 사고 차량에 박스를 그리세요</span>
                    </div>
                  )}

                  {/* 저장된 바운딩 박스 렌더링 (단일 박스) */}
                  {bboxList.map((box) => (
                    <div
                      key={box.id}
                      className="bbox-drawn"
                      style={{
                        left: box.xmin,
                        top: box.ymin,
                        width: box.xmax - box.xmin,
                        height: box.ymax - box.ymin,
                      }}
                    >
                      {isBBoxMode && (
                        <button
                          className="bbox-delete-btn"
                          onClick={(e) => {
                            e.stopPropagation();
                            setBboxList((prev) => prev.filter((b) => b.id !== box.id));
                          }}
                          title="박스 삭제"
                        >
                          ✕
                        </button>
                      )}
                    </div>
                  ))}

                  {/* 현재 드래그 중인 박스 미리보기 */}
                  {isBBoxMode && isDrawing && currentDraw && (() => {
                    const xmin = Math.min(currentDraw.startX, currentDraw.endX);
                    const ymin = Math.min(currentDraw.startY, currentDraw.endY);
                    const xmax = Math.max(currentDraw.startX, currentDraw.endX);
                    const ymax = Math.max(currentDraw.startY, currentDraw.endY);
                    return (
                      <div
                        className="bbox-preview"
                        style={{ left: xmin, top: ymin, width: xmax - xmin, height: ymax - ymin }}
                      >
                        <span className="bbox-preview-label">
                          ({xmin}, {ymin}) → ({xmax}, {ymax})
                        </span>
                      </div>
                    );
                  })()}
                </div>

                {/* 커스텀 재생바 및 이벤트 마커 표시 */}
                <div className="custom-progress-bar">
                  <div className="progress-track" onClick={handleSeek}>
                    {/* 실제 재생 진행률 */}
                    <div
                      className="progress-fill"
                      style={{
                        width: `${selectedVideo.duration ? (currentTime / selectedVideo.duration) * 100 : 0}%`,
                      }}
                    ></div>

                    {/* 이벤트 마커 (타임라인의 빨간 점) */}
                    {selectedVideo.events.map((event) => {
                      const leftPosition = (event.timestamp / selectedVideo.duration) * 100;
                      return (
                        <div
                          key={event.id}
                          className={`event-marker ${currentEventId === event.id ? 'active' : ''}`}
                          style={{ left: `${leftPosition}%` }}
                          title={event.title}
                          onClick={(e) => {
                            e.stopPropagation();
                            seekToEvent(event);
                          }}
                        />
                      );
                    })}
                  </div>
                  <div className="time-labels">
                    <span>0:00</span>
                    <span>{formatTime(selectedVideo.duration)}</span>
                  </div>
                  <div className="player-controls">
                    <button
                      className="control-btn"
                      onClick={togglePlay}
                    >
                      {isPlaying ? "⏸" : "▶"}
                    </button>

                    <label className="control-field icon-control" aria-label="Playback speed">
                      <select value={playbackSpeed} onChange={(e) => setPlaybackSpeed(e.target.value)}>
                        <option value="0.5">0.5x</option>
                        <option value="1">1x</option>
                        <option value="1.5">1.5x</option>
                        <option value="2">2x</option>
                      </select>
                    </label>

                    <label className="control-field volume-field icon-control" aria-label="Volume">
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={volume}
                        onChange={(e) => setVolume(Number(e.target.value))}
                      />
                    </label>

                    <div className="right-controls">
                      <label className="control-field quality-field icon-control" aria-label="Video quality">
                        <select value={quality} onChange={(e) => setQuality(e.target.value)}>
                          <option value="auto">Auto</option>
                          <option value="1080p">1080p</option>
                          <option value="720p">720p</option>
                          <option value="480p">480p</option>
                        </select>
                      </label>
                      <button
                        className={`control-btn theater-btn ${isTheaterMode ? "active" : ""}`}
                        onClick={() => setIsTheaterMode((enabled) => !enabled)}
                        aria-label="영화관 모드"
                      >
                        ▭
                      </button>
                      <button
                        className="control-btn fullscreen-btn"
                        onClick={handleToggleFullscreen}
                        aria-label="전체화면"
                      >
                        ⛶
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* 바운딩박스 좌표 결과 패널 */}
              {showBBoxPanel && (
                <div className="bbox-result-panel">
                  <div className="bbox-panel-header">
                    <div className="bbox-panel-title">
                      <span className="bbox-panel-icon">📦</span>
                      <span>사고 차량 바운딩박스 좌표</span>
                    </div>
                    <div className="bbox-panel-actions">
                      {bboxList.length > 0 && (
                        <button
                          className="bbox-clear-btn"
                          onClick={() => setBboxList([])}
                          title="전체 박스 초기화"
                        >
                          초기화
                        </button>
                      )}
                      <button
                        className="bbox-panel-close-btn"
                        onClick={() => setShowBBoxPanel(false)}
                        title="패널 닫기"
                      >
                        ✕
                      </button>
                    </div>
                  </div>

                  {bboxList.length === 0 ? (
                    <div className="bbox-empty-hint">
                      <span>🖱️</span>
                      <span>영상 화면에서 드래그하여 박스를 그려보세요.</span>
                    </div>
                  ) : (
                    <div className="bbox-coord-list">
                      {bboxList.map((box, idx) => (
                        <div key={box.id} className="bbox-coord-row">
                          <span className="bbox-coord-index">#{idx}</span>
                          <code className="bbox-coord-value">
                            (car, {idx}, {box.xmin}, {box.ymin}, {box.xmax}, {box.ymax})
                          </code>
                          <button
                            className="bbox-row-delete-btn"
                            onClick={() => setBboxList((prev) => prev.filter((b) => b.id !== box.id))}
                            title="삭제"
                          >
                            ✕
                          </button>
                        </div>
                      ))}
                    </div>
                  )}

                  {bboxList.length > 0 && (
                    <button
                      className="bbox-copy-btn"
                      onClick={() => {
                        const text = bboxList
                          .map((box, idx) => `(car, ${idx}, ${box.xmin}, ${box.ymin}, ${box.xmax}, ${box.ymax})`)
                          .join("\n");
                        navigator.clipboard.writeText(text).then(() => {
                          alert("좌표가 클립보드에 복사되었습니다!");
                        });
                      }}
                    >
                      📋 좌표 전체 복사
                    </button>
                  )}
                </div>
              )}
            </section>
          </div>
        </main>
      )}

      {/* 설정 모달 */}
      {isSettingsOpen && (
        <div className="settings-modal-overlay" onClick={() => setIsSettingsOpen(false)}>
          <div className="settings-modal-container" onClick={(e) => e.stopPropagation()}>
            <div className="settings-modal-sidebar">
              <h3>설정</h3>
              <button
                className={`settings-tab-btn ${activeSettingsTab === "account" ? "active" : ""}`}
                onClick={() => setActiveSettingsTab("account")}
              >
                👤 내 정보 설정
              </button>
              <button
                className={`settings-tab-btn ${activeSettingsTab === "security" ? "active" : ""}`}
                onClick={() => setActiveSettingsTab("security")}
              >
                🔑 비밀번호 변경
              </button>
              <button
                className="settings-modal-close-btn"
                onClick={() => setIsSettingsOpen(false)}
              >
                닫기
              </button>
            </div>

            <div className="settings-modal-content">
              {activeSettingsTab === "account" && (
                <div className="settings-tab-content">
                  <h2>내 정보 설정</h2>
                  <p className="tab-description">관리자 기본 정보를 확인 및 수정할 수 있습니다.</p>

                  <div className="settings-form-group">
                    <label>계정 아이디</label>
                    <input type="text" value={adminName} disabled className="disabled-input" />
                  </div>

                  <div className="settings-form-group">
                    <label>이름</label>
                    <input
                      type="text"
                      value={adminRealName}
                      onChange={(e) => setAdminRealName(e.target.value)}
                      placeholder="이름 입력"
                    />
                  </div>

                  <div className="settings-form-group">
                    <label>연락처</label>
                    <input
                      type="text"
                      value={adminPhone}
                      onChange={(e) => setAdminPhone(e.target.value)}
                      placeholder="연락처 입력"
                    />
                  </div>

                  <div className="settings-form-group">
                    <label>이메일 주소</label>
                    <input
                      type="email"
                      value={adminEmail}
                      onChange={(e) => setAdminEmail(e.target.value)}
                      placeholder="이메일 입력"
                    />
                  </div>

                  <button
                    className="settings-save-btn"
                    onClick={() => {
                      if (!adminRealName || !adminPhone || !adminEmail) {
                        alert("필수 입력 항목이 누락되었습니다.");
                        return;
                      }
                      alert("관리자 정보가 성공적으로 저장되었습니다.");
                    }}
                  >
                    수정 내용 저장
                  </button>
                </div>
              )}

              {activeSettingsTab === "security" && (
                <div className="settings-tab-content">
                  <h2>비밀번호 변경</h2>
                  <p className="tab-description">시스템 보안을 위해 주기적으로 비밀번호를 변경해 주십시오.</p>

                  <div className="settings-form-group">
                    <label>현재 비밀번호</label>
                    <input
                      type="password"
                      placeholder="현재 비밀번호 입력"
                      value={currentPassword}
                      onChange={(e) => setCurrentPassword(e.target.value)}
                    />
                  </div>

                  <div className="settings-form-group">
                    <label>새 비밀번호</label>
                    <input
                      type="password"
                      placeholder="새 비밀번호 입력"
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                    />
                  </div>

                  <div className="settings-form-group">
                    <label>새 비밀번호 확인</label>
                    <input
                      type="password"
                      placeholder="새 비밀번호 다시 입력"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                    />
                  </div>

                  <button
                    className="settings-save-btn"
                    onClick={() => {
                      if (!currentPassword || !newPassword || !confirmPassword) {
                        alert("모든 필드를 입력해 주세요.");
                        return;
                      }
                      if (newPassword !== confirmPassword) {
                        alert("새 비밀번호가 서로 일치하지 않습니다.");
                        return;
                      }
                      alert("비밀번호가 성공적으로 변경되었습니다.");
                      setCurrentPassword("");
                      setNewPassword("");
                      setConfirmPassword("");
                    }}
                  >
                    비밀번호 변경 완료
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 토스트 알림 */}
      {toast && (
        <div className={`app-toast app-toast-${toast.type}`}>{toast.message}</div>
      )}

      {/* 사고구간 CAM 클립 팝업 */}
      {clipEvent && (
        <div className="clip-modal-overlay" onClick={() => setClipEvent(null)}>
          <div className="clip-modal" onClick={(e) => e.stopPropagation()}>
            <div className="clip-modal-header">
              <span>
                사고구간 CAM 클립 · {formatTime(clipEvent.timestamp)}
                {clipEvent.endTimestamp != null
                  ? ` ~ ${formatTime(clipEvent.endTimestamp)}`
                  : ""}
                {clipEvent.prob != null
                  ? ` · ${(clipEvent.prob * 100).toFixed(1)}%`
                  : ""}
              </span>
              <button
                className="clip-modal-close"
                onClick={() => setClipEvent(null)}
              >
                ✕
              </button>
            </div>
            <video
              className="clip-modal-video"
              src={api.clipUrl(clipEvent.id)}
              controls
              autoPlay
            />
          </div>
        </div>
      )}

      {/* 영상 업로드 모달 */}
      {showUpload && (
        <UploadModal
          onClose={() => setShowUpload(false)}
          onUploaded={async () => {
            setShowUpload(false);
            showToast("업로드 완료", "success");
            await loadVideos();
          }}
          onError={(msg) => showToast(msg, "error")}
        />
      )}
    </div>
  );
}

function UploadModal({ onClose, onUploaded, onError }) {
  const [file, setFile] = useState(null);
  const [recordingDate, setRecordingDate] = useState("");
  const [uploading, setUploading] = useState(false);

  const handleUpload = async () => {
    if (!file) {
      onError("업로드할 영상 파일을 선택해 주세요.");
      return;
    }
    setUploading(true);
    try {
      await api.uploadVideo(file, recordingDate || null);
      onUploaded();
    } catch (e) {
      onError(`업로드 실패: ${e.message}`);
      setUploading(false);
    }
  };

  return (
    <div className="settings-modal-overlay" onClick={onClose}>
      <div className="upload-modal" onClick={(e) => e.stopPropagation()}>
        <h2>영상 업로드</h2>
        <p className="tab-description">분석할 CCTV 녹화 영상을 업로드합니다.</p>

        <div className="settings-form-group">
          <label>영상 파일</label>
          <input
            type="file"
            accept="video/*"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
        </div>

        <div className="settings-form-group">
          <label>녹화 일자 (달력에서 선택)</label>
          <input
            type="date"
            value={recordingDate}
            onChange={(e) => setRecordingDate(e.target.value)}
          />
        </div>

        <div className="upload-modal-actions">
          <button
            className="settings-save-btn"
            onClick={handleUpload}
            disabled={uploading}
          >
            {uploading ? "업로드 중..." : "업로드"}
          </button>
          <button className="upload-cancel-btn" onClick={onClose} disabled={uploading}>
            취소
          </button>
        </div>
      </div>
    </div>
  );
}

function RequireAuth({ children }) {
  return getToken() ? children : <Navigate to="/login" replace />;
}

export default function App() {
  const [user, setUser] = useState(() => !!getToken());

  const handleLogout = () => {
    clearAuth();
    setUser(false);
  };

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage onLogin={() => setUser(true)} />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route
          path="/videos"
          element={
            <RequireAuth>
              <Dashboard onLogout={handleLogout} view="list" />
            </RequireAuth>
          }
        />
        <Route
          path="/videos/:videoId"
          element={
            <RequireAuth>
              <Dashboard onLogout={handleLogout} view="watch" />
            </RequireAuth>
          }
        />
        <Route
          path="/analytics"
          element={
            <RequireAuth>
              <Dashboard onLogout={handleLogout} view="analytics" />
            </RequireAuth>
          }
        />
        <Route path="*" element={<Navigate to={user ? "/videos" : "/login"} replace />} />
      </Routes>
    </BrowserRouter>
  );
}
