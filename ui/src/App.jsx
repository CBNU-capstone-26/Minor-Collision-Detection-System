import { useCallback, useMemo, useRef, useState, useEffect, useSyncExternalStore } from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  useNavigate,
  useParams,
} from "react-router-dom";
import { api, saveAuth, clearAuth, getToken, getStoredUser } from "./api";
import "./App.css";
import "./incident-ui.css";
import "./premium-ui.css";
import "./dashboard-dark.css"; // 다크 모드 통일 — 반드시 premium-ui.css 다음

function InterfaceIcon({ name }) {
  const paths = {
    user: <><circle cx="12" cy="8" r="3" /><path d="M5 21v-2a7 7 0 0 1 14 0v2" /></>,
    lock: <><rect x="5" y="10" width="14" height="11" rx="3" /><path d="M8 10V7a4 4 0 0 1 8 0v3M12 14v3" /></>,
    chart: <><path d="M4 4v16h17M8 15v-4M13 15V6M18 15V9" /></>,
    moon: <path d="M20 14A8 8 0 0 1 10 4a8 8 0 1 0 10 10Z" />,
    exit: <><path d="M10 4H5v16h5M10 12h11M17 8l4 4-4 4" /></>,
  };
  return <svg className="interface-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

// 영상 카드의 ··· 메뉴 — '영상 정보 확인'은 화면 쪽 정보 팝업(onShowInfo, 업로드 일시 등 포함)을 열고,
// '영상 삭제'는 여기서 확인 창을 띄워 지운다
function VideoCardActions({ video, title, onShowInfo, onDeleted, onError }) {
  const [open, setOpen] = useState(false);
  const [dialog, setDialog] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const containerRef = useRef(null);
  const triggerRef = useRef(null);
  useEffect(() => {
    if (!open && !dialog) return;
    const outside = (event) => {
      if (!containerRef.current?.contains(event.target)) setOpen(false);
    };
    const escape = (event) => {
      if (event.key === "Escape" && !busy) {
        setOpen(false); setDialog(null); triggerRef.current?.focus();
      }
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => { document.removeEventListener("pointerdown", outside); document.removeEventListener("keydown", escape); };
  }, [open, dialog, busy]);
  const close = () => { if (!busy) { setDialog(null); triggerRef.current?.focus(); } };
  return <div className="video-card-actions" ref={containerRef} onClick={(e) => e.stopPropagation()}>
    {/* 점 세 개는 글자(···) 대신 SVG — 글꼴·OS 마다 점 위치가 달라져 처져 보이지 않게 */}
    <button className="video-more-trigger" ref={triggerRef} aria-label={`${title} 메뉴`} aria-expanded={open} onClick={() => setOpen(!open)}>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><circle cx="5" cy="12" r="2" /><circle cx="12" cy="12" r="2" /><circle cx="19" cy="12" r="2" /></svg>
    </button>
    {open && <div className="video-card-menu">
      <button onClick={() => { setOpen(false); onShowInfo(); }}><InterfaceIcon name="chart" /> 영상 정보 확인</button>
      <button className="video-menu-delete" onClick={() => { setOpen(false); setError(""); setDialog("delete"); }}><TrashIcon size={17} /> 영상 삭제</button>
    </div>}
    {dialog && <div className="settings-modal-overlay" onClick={close}>
      <div className="confirm-modal video-card-dialog" role="alertdialog" aria-modal="true" aria-label="영상 삭제 확인" onClick={(e) => e.stopPropagation()}>
        <span className="incident-eyebrow" translate="no">DELETE VIDEO</span>
        <h2>해당 영상을 삭제하시겠습니까?</h2>
        {/* 지울 영상을 카드처럼 — 썸네일 · 제목 · 녹화일자와 길이로 어떤 영상인지 바로 알아보게 */}
        <div className="video-dialog-target">
          <img className="video-dialog-thumb" src={api.thumbnailUrl(video.id)} alt="" onError={(e) => { e.currentTarget.style.visibility = "hidden"; }} />
          <div className="video-dialog-target-text">
            <strong className="video-dialog-title" title={title}>{title}</strong>
            <span className="video-dialog-meta">{[video.date, formatTime(video.duration)].filter(Boolean).join(" · ")}</span>
          </div>
        </div>
        <p className="confirm-modal-desc">원본 영상, 분석 이력과 사고 클립이 함께 삭제됩니다. 삭제 후에는 복구할 수 없습니다.</p>
        {error && <p className="video-delete-error" role="alert">{error}</p>}
        <div className="video-dialog-actions"><button autoFocus disabled={busy} onClick={close}>취소</button>
          {dialog === "delete" && <button className="video-delete-confirm" disabled={busy} onClick={async () => {
            setBusy(true); setError("");
            try { await api.deleteVideo(video.id); onDeleted(video.id); }
            catch (e) { setError(e.message); onError?.(e.message); setBusy(false); }
          }}>{busy ? "삭제 중…" : "영상 삭제"}</button>}
        </div>
      </div>
    </div>}
  </div>;
}

function TeamFooter({ animated = false }) {
  return <footer className={`siot-team-footer ${animated ? "scroll-reveal" : ""}`}>
    <div className="team-footer-brand"><strong translate="no">SIOT<span>.</span></strong><p>충북대학교 소프트웨어학부</p><small translate="no">Parking incident intelligence</small></div>
    <div className="team-footer-people"><span className="team-footer-label" translate="no">THE TEAM</span><p><span>팀장</span> 이정주</p><p><span>팀원</span> 박마누엘정, 배기원, 서범교</p></div>
    <a className="team-footer-github" href="https://github.com/CBNU-capstone-26/Minor-Collision-Detection-System" target="_blank" rel="noreferrer">GitHub <span aria-hidden="true">↗</span><small>프로젝트 저장소</small></a>
  </footer>;
}

// 선으로 그린 휴지통 아이콘 — 글자색(currentColor)을 따라가서 평소 회색 → 마우스 올리면 빨강.
// (🗑 이모지는 자체 색이라 hover 색이 안 먹고, 운영체제마다 모양이 달랐다)
function TrashIcon({ size = 14 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M3 6h18" />
      <path d="M8 6V4h8v2" />
      <path d="M19 6l-1 14H6L5 6" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
    </svg>
  );
}

// 재생 컨트롤 아이콘 — 이모지(▶ ⏸ ⛶)는 OS·글꼴마다 모양과 크기가 달라 SVG 로 그린다
const PLAYER_ICON_PATHS = {
  play: <path d="M8 5.5v13l10.5-6.5z" fill="currentColor" />,
  pause: (
    <>
      <rect x="6.5" y="5" width="3.5" height="14" rx="1" fill="currentColor" stroke="none" />
      <rect x="14" y="5" width="3.5" height="14" rx="1" fill="currentColor" stroke="none" />
    </>
  ),
  volumeHigh: (
    <>
      <path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z" fill="currentColor" />
      <path d="M15.5 9a4 4 0 0 1 0 6" />
      <path d="M18 6.5a7.5 7.5 0 0 1 0 11" />
    </>
  ),
  volumeLow: (
    <>
      <path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z" fill="currentColor" />
      <path d="M15.5 9a4 4 0 0 1 0 6" />
    </>
  ),
  volumeMute: (
    <>
      <path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z" fill="currentColor" />
      <path d="M16 9.5l5 5M21 9.5l-5 5" />
    </>
  ),
  theater: <rect x="2.5" y="6.5" width="19" height="11" rx="1.5" />,
  fullscreen: <path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5" />,
  exitFullscreen: <path d="M9 4v5H4M20 9h-5V4M15 20v-5h5M4 15h5v5" />,
};

function PlayerIcon({ name, size = 18 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {PLAYER_ICON_PATHS[name]}
    </svg>
  );
}

// 그리드의 지금 열 수 — CSS(auto-fill)가 화면 폭·배율에 맞춰 정한 값을 읽는다. el 이 바뀌거나 크기가 바뀌면 다시 읽는다
function useGridColumns(el, fallback = 3) {
  const [cols, setCols] = useState(fallback);
  useEffect(() => {
    if (!el || typeof ResizeObserver === "undefined") return undefined;
    const read = () => {
      const n = getComputedStyle(el).gridTemplateColumns.split(" ").filter(Boolean).length;
      if (n > 0) setCols(n);
    };
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  return cols;
}

// 현재 재생 시각에 표시할 사고 클립을 찾는다. 겹치는 분석이 있으면 선택된 클립, 없으면 최신 분석을 쓴다.
function clipEventAt(events, t, selectedId) {
  const inSeg = events.filter((e) => e.hasClip && e.clipStartSec != null
    && t >= e.startSec && t <= Math.max(e.startSec, e.endSec ?? e.startSec));
  return inSeg.find((e) => e.id === selectedId)
    || inSeg.reduce((best, e) => (!best || (e.analysisNo ?? 0) > (best.analysisNo ?? 0)
      || ((e.analysisNo ?? 0) === (best.analysisNo ?? 0) && e.id > best.id) ? e : best), null);
}

const EVENT_PREROLL_SEC = 1;

function clipEventsAhead(events, t, selectedId, lead) {
  const times = [t];
  events.forEach((e) => {
    if (!e.hasClip || e.clipStartSec == null) return;
    [e.startSec, Math.max(e.startSec, e.endSec ?? e.startSec) + 0.001].forEach((b) => {
      if (b > t && b <= t + lead) times.push(b);
    });
  });
  const out = [];
  times.forEach((x) => {
    const ev = clipEventAt(events, x, selectedId);
    if (ev && !out.includes(ev)) out.push(ev);
  });
  return out;
}

let toastSeq = 0; // 하단 알림 번호 (showToast)
let bboxSeq = 0; // 고른 차량 박스 번호

// 화면 너비 조건(CSS 미디어쿼리)이 지금 맞는지 — 창 크기가 바뀌면 다시 그린다
function useMediaQuery(query) {
  const subscribe = useCallback(
    (onChange) => {
      const mql = window.matchMedia(query);
      mql.addEventListener("change", onChange);
      return () => mql.removeEventListener("change", onChange);
    },
    [query]
  );
  return useSyncExternalStore(subscribe, () => window.matchMedia(query).matches);
}

// 검색 아이콘 — 색 없이 선만 있는 돋보기 (이모지 🔍 는 OS 마다 색·모양이 달라서)
function SearchIcon({ size = 16 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="11" cy="11" r="7" />
      <path d="M20 20l-4-4" />
    </svg>
  );
}

// 로그인이 만료돼 돌아왔으면(/login?expired=1) 로그인 창을 열고 안내한다
const AUTH_EXPIRED_MSG = "로그인이 만료되었습니다. 다시 로그인해 주세요.";
const cameFromExpiredLogin = () => new URLSearchParams(window.location.search).has("expired");

// 로그인·회원가입 팝업이 사라지는 효과 시간 — App.css 의 public-overlay-out·public-modal-out 과 같게
const MODAL_EXIT_MS = 200;
const prefersReducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

function LoginPage({ onLogin }) {
  const [id, setId] = useState("");
  const [pw, setPw] = useState("");
  const [error, setError] = useState(() => (cameFromExpiredLogin() ? AUTH_EXPIRED_MSG : ""));
  const [isLoginOpen, setIsLoginOpen] = useState(cameFromExpiredLogin);
  const [isSignupOpen, setIsSignupOpen] = useState(false);
  const [isModalClosing, setIsModalClosing] = useState(false);
  const modalCloseTimerRef = useRef(null);
  const navigate = useNavigate();

  // 팝업은 CSS 효과로 열리고, 닫을 땐 사라지는 효과가 끝난 뒤에 실제로 닫는다
  // ('동작 줄이기' 설정이면 효과 없이 바로 닫는다 — 그땐 CSS 효과도 꺼져 있다)
  const openLogin = () => {
    clearTimeout(modalCloseTimerRef.current);
    setIsModalClosing(false);
    setIsSignupOpen(false);
    setIsLoginOpen(true);
  };
  const closeModal = () => {
    const finish = () => {
      setIsLoginOpen(false);
      setIsSignupOpen(false);
      setIsModalClosing(false);
    };
    clearTimeout(modalCloseTimerRef.current);
    if (prefersReducedMotion()) {
      finish();
      return;
    }
    setIsModalClosing(true);
    modalCloseTimerRef.current = setTimeout(finish, MODAL_EXIT_MS);
  };

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            // Optional: unobserve if you only want it to animate once
            // observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1, rootMargin: "0px 0px -50px 0px" }
    );

    const elements = document.querySelectorAll(".scroll-reveal");
    elements.forEach((el) => observer.observe(el));

    return () => observer.disconnect();
  }, []);

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
    <div className="public-login-page">
      <header className="public-navbar">
        <button className="public-brand" onClick={() => navigate("/login")} aria-label="SIOT 홈으로 이동">
          <span className="workspace-wordmark" translate="no">SI<span className="workspace-lens">O</span>T<span className="workspace-brand-dot">.</span></span>
        </button>
        <div className="public-nav-meta">
          <nav className="premium-nav-links" aria-label="서비스 소개"><a href="#overview" translate="no">SIOT</a><a href="#why-siot">주요 기능</a><a href="#review-preview">클립 검토</a></nav>
          <button className="public-login-trigger" translate="no" onClick={openLogin}>
            LOGIN <span aria-hidden="true">↗</span>
          </button>
        </div>
      </header>

      <main className="public-main">
        <section className="public-hero" id="overview">
          <div className="public-hero-copy scroll-reveal">
            <p className="public-kicker" translate="no"><span className="premium-status-dot" /> SIOT · PARKING INTELLIGENCE</p>
            <h1>스쳐 간 순간.<br /><em>선명해지는 단서.</em></h1>
            <p className="public-hero-description">
              수많은 CCTV 기록에서, 확인이 필요한 순간으로.
              SIOT가 주차 중 접촉 의심 장면을 찾아 당신의 검토를 돕습니다.
            </p>
            <div className="public-hero-actions">
              <button className="public-primary-btn" onClick={openLogin}>
                분석 시작하기 <span aria-hidden="true">→</span>
              </button>
              <a className="public-text-link" href="#why-siot">SIOT 살펴보기 <span aria-hidden="true">↓</span></a>
            </div>
          </div>

          <div className="public-hero-panel scroll-reveal" aria-label="SIOT 분석 흐름">
            <div className="public-panel-topline" translate="no">
              <span>INCIDENT PREVIEW</span>
              <span className="public-live-dot">● DEMO</span>
            </div>
            <div className="public-scan-frame">
              <div className="public-scan-grid" />
              <div className="public-scan-road">
                <span className="public-car car-one" />
                <span className="public-car car-two" />
                <span className="public-car car-three" />
                <div className="public-detection-box box-blue-car">
                  <span className="public-detected-label" translate="no">DETECTED</span>
                </div>
              </div>
              <div className="public-scan-caption">SIMULATION / 실제 분석 결과가 아닌 예시입니다</div>
            </div>


            <div className="premium-preview-timeline" aria-hidden="true"><span /><span /><span /><i /></div>
            <div className="public-panel-footer">
              <span>의심 구간 감지</span>
              <strong>→ 날짜별 이벤트로 정리</strong>
            </div>
          </div>
        </section>

        <section className="public-proof-strip scroll-reveal" aria-label="SIOT 핵심 기능">
          <div><strong>01</strong><span>영상 업로드</span></div>
          <div><strong>02</strong><span>의심 장면 분석</span></div>
          <div><strong>03</strong><span>사건별 확인</span></div>
          <p>AI가 영상을 먼저 훑고, 관리자는 필요한 장면에 집중합니다.</p>
        </section>

        <section className="premium-showcase scroll-reveal" id="review-preview">
          <div className="premium-showcase-copy"><p className="public-kicker" translate="no">SIOT. A CLOSER LOOK.</p><h2>놓치기 쉬운 접촉.<br /><span>다시 볼 수 있는 순간.</span></h2><p>긴 영상에서 의심 구간으로.<br />클립과 원본을 함께 확인하는 명확한 검토 경험.</p><a className="public-text-link" href="#why-siot">검토 흐름 살펴보기 <span aria-hidden="true">↗</span></a></div>
          <div className="premium-review-window" aria-label="사고 의심 구간 검토 화면 예시">
            <div className="premium-window-top"><span><i /><i /><i /></span><b translate="no">SIOT / CLIP REVIEW</b><small>예시 화면</small></div>
            <div className="premium-window-scene"><div className="premium-parking-lines" /><div className="premium-scene-car parked" /><div className="premium-scene-car passing" /><div className="premium-scene-target"><span>차량 선택</span></div><span className="premium-scene-note">접촉 의심 구간 · 원본 확인 필요</span></div>
            <div className="premium-window-bottom"><span aria-hidden="true">▶</span><div><i /></div><span>00:12 / 00:48</span></div>
          </div>
        </section>
        <section className="premium-story" id="why-siot">
          <div className="premium-story-heading scroll-reveal"><p className="public-kicker" translate="no">LESS SEARCHING. MORE CLARITY.</p><h2>긴 기록은 짧게.<br /><span>중요한 순간은 깊게.</span></h2><p>차량을 고르고, 의심 구간을 찾고, 원본으로 확인하는 일.<br />복잡했던 영상 검토를 하나의 흐름으로 연결합니다.</p></div>
          <div className="premium-feature-grid">
            <article className="premium-feature scroll-reveal"><span className="premium-feature-index" translate="no">01 / FOCUS</span><div className="premium-focus-art" aria-hidden="true"><span /><i /><b /></div><h3>한 대에 집중하는 시선.</h3><p>자동 차량 탐지와 직접 지정으로<br />확인할 차량을 명확하게 선택하세요.</p><span className="premium-feature-tag">차량 탐지 · 영역 지정</span></article>
            <article className="premium-feature scroll-reveal"><span className="premium-feature-index" translate="no">02 / DISCOVER</span><div className="premium-wave-art" aria-hidden="true">{Array.from({length: 23}, (_, i) => <i key={i} style={{"--bar-height": `${20 + ((i * 37) % 65)}%`, "--bar-delay": `${i * 65}ms`}} />)}</div><h3>기록 속, 의심되는 순간.</h3><p>선택한 차량의 접촉 의심 장면을<br />분석하고 클립 단위로 살펴보세요.</p><span className="premium-feature-tag">의심 구간 분석 · 클립 확인</span></article>
            <article className="premium-feature scroll-reveal"><span className="premium-feature-index" translate="no">03 / REVIEW</span><div className="premium-review-art" aria-hidden="true"><span>00:12 <b>의심 구간</b></span><span>00:28 <b>원본 확인</b></span><span>00:41 <b>클립 검토</b></span></div><h3>판단을 위한, 더 나은 맥락.</h3><p>분석 이력과 원본 클립을 함께 확인해<br />장면의 전후 맥락을 검토하세요.</p><span className="premium-feature-tag">분석 이력 · 원본 다운로드</span></article>
          </div>
        </section>
        <section className="premium-closing scroll-reveal"><span className="public-kicker" translate="no">YOUR NEXT CLEAR MOMENT</span><h2>이제, 필요한 장면에만<br />집중하세요.</h2><button className="public-primary-btn" onClick={openLogin}>SIOT 시작하기 <span aria-hidden="true">↗</span></button><p>AI 감지는 검토를 돕는 보조 정보입니다. 최종 확인은 원본 영상으로 진행하세요.</p></section>
      </main>

      <TeamFooter animated />

      {/* 로그인·회원가입이 어두운 배경 하나를 같이 쓴다 — 서로 넘어갈 때 배경이 깜빡이지 않게 */}
      {(isLoginOpen || isSignupOpen) && (
        <div className={`login-modal-overlay ${isModalClosing ? "closing" : ""}`} onClick={closeModal}>
          {isLoginOpen ? (
            <div className="login-modal" role="dialog" aria-modal="true" aria-labelledby="login-modal-title" onClick={(e) => e.stopPropagation()}>
              <button className="login-modal-close" onClick={closeModal} aria-label="로그인 창 닫기">×</button>
              <p className="public-kicker" translate="no">SECURE ACCESS</p>
              <h2 id="login-modal-title">대시보드 로그인</h2>
              <p className="login-modal-description">분석된 영상과 사고 의심 이벤트를 확인하세요.</p>

              <div className="input-group">
                <input
                  type="text"
                  placeholder="아이디"
                  value={id}
                  onChange={(e) => setId(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleLogin()}
                  autoFocus
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
                onClick={() => {
                  setIsLoginOpen(false);
                  setIsSignupOpen(true);
                }}
              >
                회원가입
              </button>
            </div>
          ) : (
            <SignupModal onClose={closeModal} onBackToLogin={() => {
              setIsSignupOpen(false);
              setIsLoginOpen(true);
            }} />
          )}
        </div>
      )}
    </div>
  );
}

function SignupModal({ onClose, onBackToLogin }) {
  const [form, setForm] = useState({
    username: "",
    name: "",
    email: "",
    password: "",
    passwordConfirm: "",
  });
  const [error, setError] = useState("");
  const [done, setDone] = useState(false);

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
    } catch (e) {
      setError(e.message);
    }
  };

  // 어두운 배경(login-modal-overlay)은 LoginPage 가 로그인 창과 함께 쓴다
  return (
    <div className="login-modal signup-modal" role="dialog" aria-modal="true" aria-labelledby="signup-modal-title" onClick={(e) => e.stopPropagation()}>
      <button className="login-modal-close" onClick={onClose} aria-label="회원가입 창 닫기">×</button>
      <p className="public-kicker" translate="no">CREATE ACCOUNT</p>
      <h2 id="signup-modal-title">회원가입</h2>
      <p className="login-modal-description">SIOT 영상 분석 대시보드 계정을 생성합니다.</p>

      {done ? (
        <div className="signup-complete-state">
          <strong>가입이 완료되었습니다.</strong>
          <span>이제 로그인하여 대시보드를 확인할 수 있습니다.</span>
          <button className="login-submit-btn" onClick={onBackToLogin}>로그인하러 가기</button>
        </div>
      ) : (
        <>
          <div className="signup-modal-grid">
            <div className="signup-field">
              <label className="signup-field-label" htmlFor="signup-name">이름 <span>*</span></label>
              <div className="input-group">
                <input id="signup-name" type="text" placeholder="이름을 입력하세요" value={form.name} onChange={update("name")} autoFocus />
              </div>
            </div>
            <div className="signup-field">
              <label className="signup-field-label" htmlFor="signup-email">이메일</label>
              <div className="input-group">
                <input id="signup-email" type="email" placeholder="이메일을 입력하세요" value={form.email} onChange={update("email")} autoCapitalize="off" autoCorrect="off" />
              </div>
            </div>
            <div className="signup-field">
              <label className="signup-field-label" htmlFor="signup-username">아이디 <span>*</span></label>
              <div className="input-group">
                <input id="signup-username" type="text" placeholder="로그인에 사용할 아이디" value={form.username} onChange={update("username")} autoCapitalize="off" autoCorrect="off" />
              </div>
            </div>
            <div className="signup-field">
              <label className="signup-field-label" htmlFor="signup-password">비밀번호 <span>*</span></label>
              <div className="input-group">
                <input id="signup-password" type="password" placeholder="비밀번호를 입력하세요" value={form.password} onChange={update("password")} />
              </div>
            </div>
            <div className="signup-field">
              <label className="signup-field-label" htmlFor="signup-password-confirm">비밀번호 확인 <span>*</span></label>
              <div className="input-group">
                <input id="signup-password-confirm" type="password" placeholder="비밀번호를 한 번 더 입력하세요" value={form.passwordConfirm} onChange={update("passwordConfirm")} />
              </div>
            </div>
          </div>
          {error && <p className="login-error-text">{error}</p>}
          <button className="login-submit-btn" onClick={handleSignup}>가입하기</button>
          <button className="login-signup-btn" onClick={onBackToLogin}>로그인으로 돌아가기</button>
        </>
      )}
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
            <input type="text" placeholder="이름 *" value={form.name} onChange={update("name")}
              lang="ko" style={{ imeMode: "active" }} />
          </div>
          <div className="input-group">
            <input type="text" placeholder="아이디 *" value={form.username} onChange={update("username")}
              lang="en" style={{ imeMode: "inactive" }} autoCapitalize="off" autoCorrect="off" />
          </div>
          <div className="input-group">
            <input type="email" placeholder="이메일" value={form.email} onChange={update("email")}
              lang="en" style={{ imeMode: "inactive" }} autoCapitalize="off" autoCorrect="off" />
          </div>
          <div className="input-group">
            <input type="password" placeholder="비밀번호 *" value={form.password} onChange={update("password")}
              lang="en" style={{ imeMode: "inactive" }} />
          </div>
          <div className="input-group">
            <input type="password" placeholder="비밀번호 확인 *" value={form.passwordConfirm} onChange={update("passwordConfirm")}
              lang="en" style={{ imeMode: "inactive" }} />
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

// 분석이 끝나 사고 의심 클립이 만들어진 일시 — ['YYYY-MM-DD', '04시 09분'] (보는 사람 컴퓨터의 시간대)
function formatAnalyzedAt(iso) {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  const p = (n) => String(n).padStart(2, "0");
  return [`${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`, `${p(d.getHours())}시 ${p(d.getMinutes())}분`];
}

// 걸린 시간 — '45초', '3분 12초', '1시간 5분'
function formatDuration(sec) {
  if (sec == null || !Number.isFinite(sec) || sec < 0) return "";
  const s = Math.round(sec);
  if (s < 60) return `${s}초`;
  const m = Math.floor(s / 60);
  if (m < 60) return s % 60 ? `${m}분 ${s % 60}초` : `${m}분`;
  return `${Math.floor(m / 60)}시간 ${m % 60}분`;
}

const HISTORY_STATUS = {
  SUCCESS: { label: "완료", cls: "success" },
  FAILURE: { label: "실패", cls: "failure" },
  CANCELLED: { label: "취소", cls: "cancelled" },
  PENDING: { label: "대기", cls: "active" },
  PROCESSING: { label: "분석 중", cls: "active" },
};

// 영상 재생 화면 오른쪽: 이 영상의 분석 이력 (최근 분석이 위)
// history: 서버가 준 분석 이력, events: 지금 남아 있는 클립(분석 번호별로 셈),
// progressText(taskId): 진행 중인 분석의 진행 문구(없으면 null)
function AnalysisHistoryPanel({ history, events, progressText, onSelect, activeNo }) {
  const remain = {};
  events.forEach((e) => {
    if (e.analysisNo != null) remain[e.analysisNo] = (remain[e.analysisNo] || 0) + 1;
  });

  const resultOf = (h) => {
    const left = remain[h.no] || 0;
    if (h.status === "SUCCESS") {
      if (h.eventCount == null) {
        // 이력 기능 전 분석 — 찾은 수를 몰라 남은 수만 보여 준다
        return left > 0 ? { text: `사고 의심 클립 ${left}개` } : { text: "남은 클립 없음", cls: "muted" };
      }
      if (h.eventCount === 0) return { text: "사고 의심 구간 없음", cls: "muted" };
      const deleted = h.eventCount - left;
      return { text: `사고 의심 클립 ${h.eventCount}개${deleted > 0 ? ` · ${deleted}개 삭제됨` : ""}` };
    }
    if (h.status === "FAILURE") return { text: h.errorMessage || "분석 중 오류가 났습니다", cls: "error" };
    if (h.status === "CANCELLED") return { text: "사용자 요청으로 취소", cls: "muted" };
    return { text: progressText(h.taskId) || "분석 진행 중", cls: "active" };
  };

  return (
    <aside className="analysis-history">
      <div className="analysis-history-header">
        <h3>분석 이력</h3>
        <span className="analysis-history-count">총 {history.length}회</span>
      </div>
      {history.length === 0 ? (
        <div className="event-empty-hint">아직 분석한 적이 없습니다.</div>
      ) : (
        <ol className="analysis-history-list">
          {[...history].reverse().map((h) => {
            const st = HISTORY_STATUS[h.status] || { label: h.status, cls: "cancelled" };
            const at = formatAnalyzedAt(h.requestedAt);
            const took =
              h.requestedAt && h.finishedAt
                ? formatDuration((new Date(h.finishedAt) - new Date(h.requestedAt)) / 1000)
                : "";
            const result = resultOf(h);
            // 남은 클립이 있는 분석만 눌러서 클립을 볼 수 있다 — 없으면 눌러도 아무 일도 없는 일반 칸
            const hasClips = (remain[h.no] || 0) > 0;
            const body = (
              <>
                <div className="ahi-top">
                  <span className="ahi-no">분석 #{h.no}</span>
                  <span className={`ahi-status ${st.cls}`}>{st.label}</span>
                  {at && <span className="ahi-date">{at[0]}</span>}
                </div>
                {at && (
                  <div className="ahi-time">
                    {at[1]} 요청{took && ` · ${took} 소요`}
                  </div>
                )}
                <div className={`ahi-result ${result.cls || ""}`} title={result.text}>
                  {result.text}
                </div>
              </>
            );
            return (
              <li className={`analysis-history-item ${hasClips && activeNo === h.no ? "active" : ""}`} key={h.taskId}>
                {hasClips ? (
                  <button className="history-video-link" onClick={() => onSelect(h)} aria-pressed={activeNo === h.no}>
                    {body}
                    <span className="history-video-action">
                      {activeNo === h.no ? "강조 해제" : "클립 목록에서 강조"} <span aria-hidden="true">{activeNo === h.no ? "✕" : "←"}</span>
                    </span>
                  </button>
                ) : (
                  <div className="history-video-link is-static">{body}</div>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </aside>
  );
}

function getVideoDisplayTitle(video, allVideos) {
  if (!video) return "";
  if (video.title) return video.title; // 업로드할 때 넣은 제목이 있으면 그대로
  const sameDateVideos = (allVideos || []).filter((v) => v.date === video.date);
  if (sameDateVideos.length > 1) {
    const sorted = [...sameDateVideos].sort((a, b) => a.id - b.id);
    const index = sorted.findIndex((v) => v.id === video.id);
    const num = index >= 0 ? index + 1 : 1;
    return `${video.date} 녹화본 #${num}`;
  }
  return `${video.date} 녹화본`;
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

function groupVideosByDate(videos) {
  return videos.reduce((groups, video) => {
    groups[video.date] = [...(groups[video.date] ?? []), video];
    return groups;
  }, {});
}

// 대시보드 컴포넌트
function Dashboard({ onLogout, view, currentUser, onUpdateUser }) {
  const navigate = useNavigate();
  const { videoId } = useParams();

  // 상태 관리
  const [videos, setVideos] = useState([]); // API에서 로드한 영상 목록
  const [isLoadingVideos, setIsLoadingVideos] = useState(false);
  const [videoLoadError, setVideoLoadError] = useState(null);
  const [filterDays, setFilterDays] = useState(9999); // 기본 전체
  const [searchQuery, setSearchQuery] = useState(""); // 상단 검색어

  const [selectedVideo, setSelectedVideo] = useState(null); // null이면 홈(그리드) 화면, 값이 있으면 영상 재생 화면
  const [currentEventId, setCurrentEventId] = useState(null); // 현재 선택된 이벤트 마커
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0); // 실제 영상 재생 위치(초)
  const [selectedCalendarDate, setSelectedCalendarDate] = useState(null); // 달력에서 선택한 날짜
  const [datePickerVideos, setDatePickerVideos] = useState(null); // 동일 날짜 영상 복수 존재 시 선택 모달 데이터
  const [playbackSpeed, setPlaybackSpeed] = useState("1");
  const [volume, setVolume] = useState(70);
  const [isMuted, setIsMuted] = useState(false);
  const [mediaDuration, setMediaDuration] = useState(0); // <video> 가 알려준 실제 길이(초, 소수) — 재생바 위치 계산용
  const [isSeeking, setIsSeeking] = useState(false); // 재생바를 누른 채 끌어서 탐색 중
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isTheaterMode, setIsTheaterMode] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(true); // 오른쪽 패널(진행 단계 + 분석 이력) — 기본은 열림(영상을 열 때마다)
  const [highlightAnalysisNo, setHighlightAnalysisNo] = useState(null); // 분석 이력에서 고른 분석 — 그 클립을 왼쪽 목록에서 강조
  const [clipView, setClipView] = useState("cam");
  const camChipRef = useRef(null);
  const eventListRef = useRef(null);
  const [calendarMonth, setCalendarMonth] = useState(new Date("2026-05-01T00:00:00"));
  const playerContainerRef = useRef(null);
  const seekHoverRef = useRef(null); // 재생바 위 마우스 위치의 시각 말풍선 (마우스 움직임마다 다시 그리지 않게 직접 갱신)
  const isSeekingRef = useRef(false); // 끄는 중인지 — 이벤트 처리 중엔 다시 그리기 전이라 상태 대신 이걸 본다
  const detectAbortControllerRef = useRef(null);
  const analyzeAbortControllerRef = useRef(null);

  // 바운딩박스 / 차량 지정 모드 상태 ("none" | "manual" | "auto")
  const [selectionMode, setSelectionMode] = useState("none");
  const [isBBoxMode, setIsBBoxMode] = useState(false);

  const [bboxList, setBboxList] = useState([]);
  const [currentDraw, setCurrentDraw] = useState(null); // { startX, startY, endX, endY }
  const [isDrawing, setIsDrawing] = useState(false);
  const bboxOverlayRef = useRef(null);
  const videoElRef = useRef(null); // 실제 <video> 엘리먼트 (원본 해상도 환산용)

  // YOLO 차량 자동 탐지 및 마우스 Hover 관련 상태
  const [isDetectingVehicles, setIsDetectingVehicles] = useState(false);
  const [detectedBoxes, setDetectedBoxes] = useState([]); // [{ id, class_name, confidence, bbox: [x1,y1,x2,y2] }]
  const [hoveredDetectedBox, setHoveredDetectedBox] = useState(null);


  // 사고감지 실행 / 분석 관련 상태
  const [showAnalyzeConfirm, setShowAnalyzeConfirm] = useState(false); // 확인 팝오버
  const [toast, setToast] = useState(null); // { type, message }
  // 동시 분석 작업 목록: [{ taskId, videoId, dateLabel, estimatedSec, startedAt }]
  const [analyzingJobs, setAnalyzingJobs] = useState([]);
  const [showJobsDropdown, setShowJobsDropdown] = useState(false);
  const [now, setNow] = useState(() => Date.now()); // 진행률 계산용 현재시각(0.5s마다 갱신)
  const selectedVideoIdRef = useRef(null); // 폴링 중 최신 선택 영상 추적(stale closure 방지)
  const progressSmoothRef = useRef({}); // 작업별 화면 진행률(부드럽게 움직이고 뒤로 가지 않게)
  const aliveRef = useRef(true); // 화면이 떠 있는 동안만 상태 폴링을 이어 간다
  const pollingRef = useRef(new Set()); // 이미 상태를 확인 중인 작업(중복 폴링 방지)
  const [clipEvent, setClipEvent] = useState(null); // 사고 클립 팝업 대상 이벤트
  const [pendingDeleteEvent, setPendingDeleteEvent] = useState(null); // 삭제 확인 창 대상 이벤트
  const [pendingClearAll, setPendingClearAll] = useState(false); // 전체 삭제 확인 창
  const [infoVideo, setInfoVideo] = useState(null); // 영상 목록 카드의 ⋮ → 영상 정보 팝업 대상

  // 업로드 모달
  const [showUpload, setShowUpload] = useState(false);

  // 영상 삭제 모드
  const [deleteMode, setDeleteMode] = useState(false);
  const [selectedForDelete, setSelectedForDelete] = useState([]); // 선택된 영상 id 목록

  const toggleDeleteSelect = (videoId) => {
    setSelectedForDelete((prev) =>
      prev.includes(videoId)
        ? prev.filter((id) => id !== videoId)
        : [...prev, videoId]
    );
  };

  const exitDeleteMode = () => {
    setDeleteMode(false);
    setSelectedForDelete([]);
  };

  const handleDeleteSelected = async () => {
    if (selectedForDelete.length === 0) {
      showToast("삭제할 영상을 선택해 주세요.", "warning");
      return;
    }
    if (!window.confirm(`선택한 ${selectedForDelete.length}개 영상을 삭제하시겠습니까? 되돌릴 수 없습니다.`)) {
      return;
    }
    try {
      await Promise.all(selectedForDelete.map((id) => api.deleteVideo(id)));
      showToast(`${selectedForDelete.length}개 영상을 삭제했습니다.`, "success");
      exitDeleteMode();
      await loadVideos();
    } catch (e) {
      showToast(`삭제 실패: ${e.message}`, "error");
    }
  };

  // 하단 알림. 알림마다 번호를 붙여, 타이머는 자기 알림이 아직 떠 있을 때만 닫는다
  // (앞 알림의 타이머가 뒤에 뜬 알림 — 예: 7초짜리 분석 결과 — 을 일찍 닫지 않게)
  const showToast = (message, type = "warning", durationMs = 2600) => {
    const id = ++toastSeq;
    setToast({ id, message, type });
    setTimeout(() => setToast((t) => (t && t.id === id ? null : t)), durationMs);
  };

  // 프로필 & 설정 관련 상태
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [activeSettingsTab, setActiveSettingsTab] = useState("account"); // "account", "security"

  // 설정 정보 (현재 로그인 사용자 정보와 동기화)
  const currentUsername = currentUser?.username || "user";
  const [profileName, setProfileName] = useState(currentUser?.name || "");
  const [profilePhone, setProfilePhone] = useState(currentUser?.phone || "");
  const [profileEmail, setProfileEmail] = useState(currentUser?.email || "");

  useEffect(() => {
    if (currentUser) {
      setProfileName(currentUser.name || "");
      setProfilePhone(currentUser.phone || "");
      setProfileEmail(currentUser.email || "");
    }
  }, [currentUser]);

  const displayName = currentUser?.name || currentUser?.username || "사용자";
  const avatarInitial = (displayName[0] || currentUsername[0] || "U").toUpperCase();
  const roleLabel = currentUser?.role === "ADMIN" ? "시스템 관리자" : "일반 사용자";

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
    setIsLoadingVideos(true);
    setVideoLoadError(null);
    try {
      const list = await api.listVideos(9999);
      setVideos(list);
      return list;
    } catch (e) {
      const message = `영상 목록을 불러오지 못했습니다: ${e.message}`;
      setVideoLoadError(message);
      showToast(message, "error");
      return [];
    } finally {
      setIsLoadingVideos(false);
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
        setIsHistoryOpen(true);
        setHighlightAnalysisNo(null);
        setBboxList([]);
        setIsBBoxMode(false);
        setSelectionMode("none");
        setShowAnalyzeConfirm(false);

        setCurrentTime(0);
        setMediaDuration(0);
        if (v.detected_vehicles) {
          try {
            setDetectedBoxes(JSON.parse(v.detected_vehicles));
          } catch {
            setDetectedBoxes([]);
          }
        } else {
          setDetectedBoxes([]);
        }
        setHoveredDetectedBox(null);

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

  // 기간 필터만 적용한 목록
  const periodVideos = useMemo(() => {
    if (filterDays >= 9999) return videos;
    const startDate = formatDate(addDays(new Date(), -filterDays));
    return videos.filter((video) => video.date >= startDate);
  }, [filterDays, videos]);

  // 기간 + 검색어 필터링 — 영상 목록 페이지용
  const filteredVideos = useMemo(() => {
    let list = periodVideos;

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      // 제목은 카드에 보이는 그대로('2026-10-12 녹화본 #5')로 찾는다
      list = list.filter(
        (v) =>
          (v.date || "").toLowerCase().includes(q) ||
          getVideoDisplayTitle(v, videos).toLowerCase().includes(q) ||
          (v.events && v.events.some((e) => (e.title || "").toLowerCase().includes(q)))
      );
    }

    // 녹화 일자 최신순 정렬 (같은 날짜는 나중에 올린 영상 먼저)
    return [...list].sort((a, b) => {
      if (a.date !== b.date) return (b.date || "").localeCompare(a.date || "");
      return b.id - a.id;
    });
  }, [periodVideos, searchQuery, videos]);

  // 메인화면 영상 목록 페이지네이션 — 한 페이지 = 지금 열 수 × 줄 수(두 줄 이상, 최소 6개).
  // 3열 이하(보통 화면)는 예전처럼 6개, 화면이 넓거나 축소해 열이 늘면 줄을 꽉 채운다(예: 6열 → 12개)
  const [videoGridEl, setVideoGridEl] = useState(null);
  const gridCols = useGridColumns(videoGridEl);
  const VIDEOS_PER_PAGE = gridCols * Math.max(2, Math.ceil(6 / gridCols));
  const [currentPage, setCurrentPage] = useState(1);

  const totalPages = Math.max(1, Math.ceil(filteredVideos.length / VIDEOS_PER_PAGE));

  // 한 페이지 개수가 바뀌면(창 크기·배율 변경) 보던 첫 영상이 들어 있는 페이지로 옮긴다
  const prevPerPage = useRef(VIDEOS_PER_PAGE);
  useEffect(() => {
    if (prevPerPage.current === VIDEOS_PER_PAGE) return;
    const firstIndex = (currentPage - 1) * prevPerPage.current;
    prevPerPage.current = VIDEOS_PER_PAGE;
    setCurrentPage(Math.floor(firstIndex / VIDEOS_PER_PAGE) + 1);
  }, [VIDEOS_PER_PAGE, currentPage]);

  // 필터 조건(기간, 검색어) 변경 시 1페이지로 리셋
  useEffect(() => {
    setCurrentPage(1);
  }, [filterDays, searchQuery]);

  // 영상 삭제 등으로 인해 현재 페이지가 전체 페이지 수보다 커지면 조정
  useEffect(() => {
    if (currentPage > totalPages) {
      setCurrentPage(totalPages);
    }
  }, [totalPages, currentPage]);

  const paginatedVideos = useMemo(() => {
    const startIndex = (currentPage - 1) * VIDEOS_PER_PAGE;
    return filteredVideos.slice(startIndex, startIndex + VIDEOS_PER_PAGE);
  }, [filteredVideos, currentPage, VIDEOS_PER_PAGE]);

  const handlePageChange = (newPage) => {
    setCurrentPage(newPage);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const getPageNumbers = () => {
    const maxButtons = 5;
    let start = Math.max(1, currentPage - 2);
    let end = Math.min(totalPages, start + maxButtons - 1);
    if (end - start + 1 < maxButtons) {
      start = Math.max(1, end - maxButtons + 1);
    }
    const pages = [];
    for (let i = start; i <= end; i++) {
      pages.push(i);
    }
    return pages;
  };

  const videosByDate = useMemo(() => {
    return groupVideosByDate(videos);
  }, [videos]);

  const calendarDays = useMemo(() => getCalendarDays(calendarMonth), [calendarMonth]);

  const moveCalendarMonth = (offset) => {
    setCalendarMonth((month) => new Date(month.getFullYear(), month.getMonth() + offset, 1));
  };

  const handleCalendarDateClick = (date) => {
    const dateText = formatDate(date);
    setSelectedCalendarDate(dateText); // 영상 유무와 무관하게 날짜 선택 표시
    const dayVideos = videosByDate[dateText] ?? [];
    if (dayVideos.length === 1) {
      handleWatchVideo(dayVideos[0]);
    } else if (dayVideos.length > 1) {
      setDatePickerVideos({ dateText, videos: dayVideos });
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

  // 특정 사고 이벤트 1건 삭제
  // 휴지통 → 바로 지우지 않고 화면 가운데 확인 창을 띄운다(서버가 클립 파일도 지움)
  const requestDeleteEvent = (event, e) => {
    if (e) e.stopPropagation();
    setPendingDeleteEvent(event);
  };

  const handleDeleteEvent = async (eventId) => {
    if (!selectedVideo) return;
    try {
      await api.deleteEvent(selectedVideo.id, eventId);
      setSelectedVideo((prev) =>
        prev
          ? {
            ...prev,
            events: prev.events.filter((ev) => ev.id !== eventId),
          }
          : null
      );
      setVideos((prev) =>
        prev.map((v) =>
          v.id === selectedVideo.id
            ? { ...v, events: v.events.filter((ev) => ev.id !== eventId) }
            : v
        )
      );
      if (currentEventId === eventId) {
        setCurrentEventId(null);
      }
      showToast("감지 이벤트가 삭제되었습니다.", "info");
    } catch (err) {
      showToast(`이벤트 삭제 실패: ${err.message}`, "error");
    }
  };

  // 영상의 모든 사고 이벤트 전체 삭제 — 개별 삭제와 같은 확인 창을 거친다
  const requestClearAllEvents = () => {
    if (!selectedVideo || selectedVideo.events.length === 0) return;
    setPendingClearAll(true);
  };

  const handleClearAllEvents = async () => {
    if (!selectedVideo || selectedVideo.events.length === 0) return;
    try {
      await api.clearEvents(selectedVideo.id);
      setSelectedVideo((prev) => (prev ? { ...prev, events: [] } : null));
      setVideos((prev) =>
        prev.map((v) => (v.id === selectedVideo.id ? { ...v, events: [] } : v))
      );
      setCurrentEventId(null);
      showToast("모든 감지 이벤트가 삭제되었습니다.", "info");
    } catch (err) {
      showToast(`이벤트 전체 삭제 실패: ${err.message}`, "error");
    }
  };


  // 수동 차량 지정 모드 시작
  const handleStartManualDesignate = () => {
    // 차량 탐지처럼 영상을 멈춰, 지금 보이는 화면에서 차량을 그리게 한다
    if (videoElRef.current) {
      videoElRef.current.pause();
      setIsPlaying(false);
    }
    setSelectionMode("manual");
    setIsBBoxMode(true);
    setHoveredDetectedBox(null);
    // 안내 문구는 잠깐 뜨는 알림 대신 버튼 위 말풍선으로 모드가 끝날 때까지 띄운다 (.designate-hint)
  };

  // 탐지/지정 모드 해제
  const handleEndDesignateMode = () => {
    setSelectionMode("none");
    setIsBBoxMode(false);
    setCurrentDraw(null);
    setIsDrawing(false);
    setHoveredDetectedBox(null);
  };

  // YOLO 차량 탐지 취소
  const handleCancelDetectVehicles = () => {
    if (detectAbortControllerRef.current) {
      detectAbortControllerRef.current.abort();
      detectAbortControllerRef.current = null;
    }
    setIsDetectingVehicles(false);
    setSelectionMode("none");
    showToast("차량 탐지가 취소되었습니다.", "info");
  };

  // YOLO 차량 탐지 실행 (영상 pause → 백엔드 탐지 → Hover 선택 활성화)
  const handleAutoDetectVehicles = async () => {
    if (!selectedVideo) return;
    if (videoElRef.current) {
      videoElRef.current.pause();
      setIsPlaying(false);
    }
    setSelectionMode("auto");
    setIsDetectingVehicles(true);   // 진행 표시는 영상 위 로딩으로 한다(알림 없음)

    const controller = new AbortController();
    detectAbortControllerRef.current = controller;

    try {
      const request = await api.detectVehicles(selectedVideo.id, currentTime, controller.signal);
      let result;
      while (!result) {
        if (controller.signal.aborted) {
          throw new DOMException("차량 탐지가 취소되었습니다.", "AbortError");
        }
        const status = await api.getVehicleDetectionTask(
          selectedVideo.id, request.task_id, controller.signal,
        );
        if (status.status === "SUCCESS") {
          result = status;
          break;
        }
        if (status.status === "FAILURE" || status.status === "CANCELLED") {
          throw new Error(status.error_message || "차량 탐지에 실패했습니다.");
        }
        await new Promise((resolve) => setTimeout(resolve, 700));
      }
      const list = result.detected_vehicles || [];
      setDetectedBoxes(list);
      setIsBBoxMode(true);
      // 탐지 성공은 알리지 않는다(영상 위에 차량이 바로 표시됨). 한 대도 없을 때만 다음 행동을 안내한다.
      if (list.length === 0) {
        showToast("탐지된 차량이 없습니다. 직접 차량 지정을 이용해 주세요.", "warning");
      }
    } catch (err) {
      if (err.name === "AbortError") {
        return;
      }
      showToast(`차량 탐지 실패: ${err.message}`, "error");
    } finally {
      setIsDetectingVehicles(false);
      detectAbortControllerRef.current = null;
    }
  };



  // '사고 구간 찾기' 클릭 → 차량 박스가 없으면 경고, 있으면 확인 팝오버
  const handleAnalyzeClick = () => {
    if (bboxList.length === 0) {
      showToast("사고를 감지할 차량을 드래그하여 주십시오.", "warning");
      return;
    }
    setShowAnalyzeConfirm(true);
  };

  // 사고감지 분석 작업 취소
  // 보고 있는 영상의 상세(분석 이력 포함)를 다시 받아 온다 — 그사이 다른 영상으로 옮겼으면 덮어쓰지 않는다
  const refreshSelectedVideo = async (videoId) => {
    if (videoId == null || selectedVideoIdRef.current !== videoId) return;
    try {
      const v = await api.getVideo(videoId);
      if (selectedVideoIdRef.current === videoId) setSelectedVideo(v);
    } catch {
      /* 이력 갱신에 실패해도 다음 갱신 때 맞춰진다 */
    }
  };

  const handleCancelAnalysis = async (taskId) => {
    if (analyzeAbortControllerRef.current) {
      analyzeAbortControllerRef.current.abort();
      analyzeAbortControllerRef.current = null;
    }
    if (taskId) {
      try {
        const cancelledVideoId = analyzingJobs.find((j) => j.taskId === taskId)?.videoId;
        await api.cancelTask(taskId);
        removeJob(taskId);
        showToast("사고 감지 분석 작업을 취소했습니다.", "info");
        refreshSelectedVideo(cancelledVideoId); // 분석 이력에 '취소' 반영
        loadVideos();
      } catch (e) {
        showToast(`분석 취소 실패: ${e.message}`, "error");
      }
    } else {
      showToast("분석 요청이 취소되었습니다.", "info");
    }
  };

  // 분석 태스크 상태 폴링 (videoId = 분석을 시작한 영상 id)
  const removeJob = (taskId) =>
    setAnalyzingJobs((prev) => prev.filter((j) => j.taskId !== taskId));

  // 분석이 끝났을 때: 화면 안 토스트 + 다른 탭·창을 보고 있으면 브라우저 알림
  const notifyDone = (title, body, type) => {
    showToast(`${title}: ${body}`, type, 7000); // 분석 결과는 놓치지 않게 7초
    try {
      if (document.hidden && "Notification" in window && Notification.permission === "granted") {
        new Notification(title, { body, icon: "/siot-logo.svg" });
      }
    } catch {
      /* 브라우저 알림을 못 띄워도 토스트는 이미 표시됨 */
    }
  };

  const pollTask = (taskId, videoId) => {
    if (pollingRef.current.has(taskId)) return; // 같은 작업을 두 번 폴링하지 않는다
    pollingRef.current.add(taskId);
    const finish = () => {
      pollingRef.current.delete(taskId);
      removeJob(taskId);
    };
    const poll = async () => {
      if (!aliveRef.current) return; // 화면이 닫히면 멈춤 — 다시 열리면 복원하면서 새로 시작
      try {
        const status = await api.getTaskStatus(taskId);
        if (!aliveRef.current) return;
        if (status.status === "SUCCESS") {
          finish();
          notifyDone("분석 완료", `사고 의심 구간 ${status.events.length}건`, "success");
          // 분석한 영상을 아직 보고 있을 때만 화면 갱신 (다른 영상으로 이동했으면 덮어쓰지 않음)
          if (selectedVideoIdRef.current === videoId) {
            const v = await api.getVideo(videoId);
            setSelectedVideo(v);
          }
          loadVideos();
          return;
        }
        if (status.status === "FAILURE") {
          finish();
          notifyDone("분석 실패", status.error_message || "오류", "error");
          if (selectedVideoIdRef.current === videoId) {
            const v = await api.getVideo(videoId); // '다시 실행' 안내를 띄우기 위해 갱신
            setSelectedVideo(v);
          }
          return;
        }
        if (status.status === "CANCELLED") {
          finish();
          showToast("사고 감지 분석이 취소되었습니다.", "info");
          refreshSelectedVideo(videoId);
          return;
        }
        // 진행도(%)와 서버가 실제로 잰 속도 기준의 남은 시간(초)을 반영한다.
        // 받은 시각(etaAt)을 같이 저장해 다음 확인 전까지 화면에서 1초씩 줄여 보여 준다.
        const receivedAt = Date.now();
        setAnalyzingJobs((prev) =>
          prev.map((j) => {
            if (j.taskId !== taskId) return j;
            const next = { ...j, status: status.status };
            if (typeof status.progress === "number") {
              next.progress = status.progress;
              next.progressAt = receivedAt;             // 받은 시각 — 진행 속도 계산용
            }
            next.etaSec = status.eta_sec ?? null;
            next.etaMin = !!status.eta_is_min;
            next.etaAt = receivedAt;
            next.phase = status.stage ?? null;          // 지금 단계(훑어보기 등)
            next.phaseCount = status.stage_count ?? null;
            next.phaseIndex = status.stage_index ?? null;
            return next;
          }));
        setTimeout(poll, 2000); // PENDING/PROCESSING → 재시도
      } catch (e) {
        finish();
        showToast(`상태 조회 실패: ${e.message}`, "error");
      }
    };
    poll();
  };

  // 실패·중단된 마지막 분석을 같은 차량 박스로 다시 실행
  const handleRetryAnalysis = async () => {
    const last = selectedVideo?.lastTask;
    if (!last) return;
    const videoIdForJob = selectedVideo.id;
    try {
      const res = await api.retryTask(last.id);
      const job = {
        taskId: res.task_id,
        videoId: videoIdForJob,
        dateLabel: selectedVideo.date,
        status: res.status,
      };
      showToast(res.already_running ? "이 영상은 이미 분석 중입니다." : "분석을 다시 시작했습니다.", "info");
      setAnalyzingJobs((prev) => (prev.some((j) => j.taskId === res.task_id) ? prev : [...prev, job]));
      pollTask(res.task_id, videoIdForJob);
      setSelectedVideo((v) =>
        v ? { ...v, lastTask: { id: res.task_id, status: res.status, errorMessage: null } } : v);
      refreshSelectedVideo(videoIdForJob); // 분석 이력에 새 분석 추가
    } catch (e) {
      showToast(`다시 실행 실패: ${e.message}`, "error");
    }
  };

  // 화면이 열릴 때(새로고침·재접속 포함) 서버에 남아 있는 내 분석을 복원해 이어서 추적한다.
  // 분석 목록은 브라우저 메모리에만 있어서, 이게 없으면 창을 닫았다 열었을 때 진행 중인
  // 분석이 사라진 것처럼 보이고 같은 영상을 다시 요청하게 된다.
  useEffect(() => {
    aliveRef.current = true;
    (async () => {
      try {
        const tasks = await api.listActiveTasks();
        if (!aliveRef.current) return;
        tasks.forEach((t) => {
          const job = {
            taskId: t.task_id,
            videoId: t.video_id,
            dateLabel: t.recording_date || (t.created_at || "").slice(0, 10),
            startedAt: Date.now(),
            status: t.status,
            progress: t.progress ?? undefined,
            etaSec: t.eta_sec ?? null,
            etaMin: !!t.eta_is_min,
            etaAt: Date.now(),
            phase: t.stage ?? null,
            phaseCount: t.stage_count ?? null,
            phaseIndex: t.stage_index ?? null,
            progressAt: Date.now(),
          };
          setAnalyzingJobs((prev) => (prev.some((j) => j.taskId === t.task_id) ? prev : [...prev, job]));
          pollTask(t.task_id, t.video_id);
        });
      } catch {
        /* 복원 실패는 치명적이지 않다 — 새 분석 요청은 그대로 동작 */
      }
    })();
    return () => {
      aliveRef.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 확인 팝오버에서 '실행' → bbox 원본해상도 환산 후 분석 요청
  const runAnalysis = async () => {
    setShowAnalyzeConfirm(false);
    const box = bboxList[0];
    if (!box) return;

    // bboxList 는 원본 영상 해상도 좌표 — 지금 화면 크기와 상관없이 그대로 보낸다
    const bbox = {
      bbox_xmin: box.xmin,
      bbox_ymin: box.ymin,
      bbox_xmax: box.xmax,
      bbox_ymax: box.ymax,
    };

    // 분석 시작: 지정 모드를 끄고 해당 영상을 '분석 중'으로 표시
    // (박스 그리기 판만 끄면 모드 값이 남아 분석 중에도 탐지 버튼이 켜진 상태로 보였다)
    const videoId = selectedVideo.id;
    handleEndDesignateMode();

    // 예상 소요시간(초) 추정: 슬라이딩 윈도우 수(총프레임/stride) 기반 (CPU ~1s/윈도우)
    // ※ 분모는 백엔드 config.PREDICT_WINDOW_STRIDE(현재 5)와 반드시 일치시킬 것
    const totalFrames = Math.round(
      (selectedVideo.duration || 0) * (selectedVideo.fps || 30));
    const estimatedSec = Math.max(10, Math.round((totalFrames / 5) * 1.1));
    const job = {
      taskId: null,
      videoId,
      dateLabel: selectedVideo.date,
      estimatedSec,
      startedAt: Date.now(),
    };

    // 분석이 끝나면 다른 탭에 있어도 알려 주도록 처음 한 번 알림 권한을 묻는다
    try {
      if ("Notification" in window && Notification.permission === "default") {
        const r = Notification.requestPermission();
        if (r && r.catch) r.catch(() => {});
      }
    } catch {
      /* 알림을 지원하지 않는 브라우저 — 토스트만 쓴다 */
    }

    const controller = new AbortController();
    analyzeAbortControllerRef.current = controller;

    try {
      const res = await api.analyze(videoId, bbox, controller.signal);
      job.taskId = res.task_id;
      job.status = res.status;
      if (res.already_running) {
        // 같은 영상에 이미 대기·진행 중인 분석이 있으면 서버가 새로 만들지 않고 그 작업을 준다
        showToast("이 영상은 이미 분석 중입니다. 진행 상황을 이어서 보여드릴게요.", "info");
      }
      setAnalyzingJobs((prev) => (prev.some((j) => j.taskId === res.task_id) ? prev : [...prev, job]));
      pollTask(res.task_id, videoId);
      refreshSelectedVideo(videoId); // 분석 이력에 새 분석 추가
    } catch (e) {
      if (e.name === "AbortError") {
        return;
      }
      showToast(`분석 요청 실패: ${e.message}`, "error");
    } finally {
      analyzeAbortControllerRef.current = null;
    }
  };

  const handleToggleFullscreen = () => {
    const player = playerContainerRef.current;
    if (!player) return;

    if (document.fullscreenElement) {
      document.exitFullscreen();
      return;
    }

    if (player.requestFullscreen) {
      player.requestFullscreen();
    } else if (player.webkitRequestFullscreen) {
      player.webkitRequestFullscreen(); // 옛 Safari
    } else {
      // iPhone Safari 는 영상 영역(div) 전체화면을 허용하지 않는다 → 영상만이라도 기기 전체화면으로
      videoElRef.current?.webkitEnterFullscreen?.();
    }
  };

  // ── 실제 <video> 재생 제어 ──────────────────────────────
  // 차량 탐지 중엔 재생·이동을 막는다 — 탐지는 요청한 순간의 화면으로 하므로 화면이 바뀌면 박스가 어긋난다
  const togglePlay = () => {
    const v = videoElRef.current;
    if (!v || isDetectingVehicles) return;
    if (v.paused) v.play();
    else v.pause();
  };

  // 재생바 길이 기준 — 실제 영상 길이(소수)를 쓰고, 영상 정보를 받기 전엔 서버가 준 길이(정수 초)
  const playerDuration = mediaDuration || selectedVideo?.duration || 0;

  // 재생바: 누르거나 누른 채 끌어서 탐색하고, 마우스를 올리면 그 위치의 시각을 말풍선으로 보여준다
  const seekRatioAt = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  };

  const seekToRatio = (ratio) => {
    const v = videoElRef.current;
    if (!v || !v.duration) return;
    v.currentTime = ratio * v.duration;
    setCurrentTime(v.currentTime);
  };

  const updateSeekBubble = (e, ratio) => {
    const bubble = seekHoverRef.current;
    if (!bubble) return;
    bubble.textContent = formatTime(Math.floor(ratio * playerDuration));
    // 양 끝에서 말풍선이 재생바 밖으로 잘리지 않게 안쪽으로 붙인다
    const width = e.currentTarget.getBoundingClientRect().width;
    const half = bubble.offsetWidth / 2;
    bubble.style.left = `${Math.min(width - half, Math.max(half, ratio * width))}px`;
  };

  const handleSeekStart = (e) => {
    if (e.button !== 0 || isDetectingVehicles) return;
    e.currentTarget.setPointerCapture?.(e.pointerId);
    isSeekingRef.current = true;
    setIsSeeking(true);
    const ratio = seekRatioAt(e);
    updateSeekBubble(e, ratio);
    seekToRatio(ratio);
  };

  const handleSeekMove = (e) => {
    const ratio = seekRatioAt(e);
    updateSeekBubble(e, ratio);
    if (isSeekingRef.current) seekToRatio(ratio);
  };

  const handleSeekEnd = (e) => {
    if (!isSeekingRef.current) return;
    isSeekingRef.current = false;
    if (e.currentTarget.hasPointerCapture?.(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    setIsSeeking(false);
  };

  // 이벤트 구간 클릭 → 1초 앞부터 재생해 사고 직전 상황도 볼 수 있게 한다.
  const seekToEvent = (event) => {
    if (isDetectingVehicles) return;
    setCurrentEventId(event.id);
    const v = videoElRef.current;
    if (v) {
      v.currentTime = Math.max(0, (event.startSec ?? event.timestamp) - EVENT_PREROLL_SEC);
      v.play().catch(() => {});
    }
  };

  const toggleMute = () => {
    if (isMuted || volume === 0) {
      if (volume === 0) setVolume(50);
      setIsMuted(false);
    } else {
      setIsMuted(true);
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

  useEffect(() => {
    const v = videoElRef.current;
    if (v) v.muted = isMuted;
  }, [isMuted, selectedVideo]);

  // 영상 칸 크기가 바뀌면(창 크기 · 안내 바 · 실패 안내 · 전체화면) 차량 박스 위치를 다시 계산하도록 다시 그린다.
  // 영상 칸은 화면 높이에 맞춰 줄어들 수 있어, 렌더 시점의 크기로 그린 박스가 어긋나지 않게 한다.
  const [overlaySize, setOverlaySize] = useState(null); // { w, h } — 박스 그리기용(렌더 중 ref 를 읽지 않게 상태로 둔다)
  useEffect(() => {
    const el = bboxOverlayRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      const { width, height } = el.getBoundingClientRect();
      setOverlaySize((prev) => (prev && prev.w === width && prev.h === height ? prev : { w: width, h: height }));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [selectedVideo?.id, view]);

  // 분석 이력에서 분석을 고르면, 그 분석의 첫 클립이 보이도록 왼쪽 목록을 스크롤한다
  useEffect(() => {
    if (highlightAnalysisNo == null) return;
    const el = eventListRef.current?.querySelector(".event-item.analysis-highlight");
    el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [highlightAnalysisNo]);

  // 전체화면 버튼 아이콘을 실제 상태에 맞춘다 (Esc 로 나가도 따라가게)
  useEffect(() => {
    const onChange = () =>
      setIsFullscreen(!!document.fullscreenElement && document.fullscreenElement === playerContainerRef.current);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

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

  // 작업별 진행률(%)과 단계 문구.
  // 백엔드가 디코딩·선별·추론·렌더링 전 단계에서 '실제' 진행도와 지금 단계를 보고하므로
  // 가짜 시간기반 추정을 만들지 않는다. 남은 시간은 단계가 바뀔 때 숫자가 뛰어 혼란스러워서
  // 보여 주지 않고, 대신 지금 무엇을 하는지(단계)와 부드럽게 움직이는 진행률을 보여 준다.

  // 분석 단계 문구 — 서버(model/inference.py _report, backend prediction_job)의 stage 값.
  // main 은 '~하는 중'(뒤에 움직이는 점이 붙음), extra 는 그 뒤에 붙는 보조 정보.
  const PHASE_TEXT = {
    loading: () => ({ main: "분석 엔진 준비 중" }),
    decode: () => ({ main: "분석 엔진 준비 중" }),   // 영상 읽기도 준비 과정으로 묶어 따로 알리지 않는다
    motion: () => ({ main: "차량 움직임 계산 중" }),
    coarse: () => ({ main: "영상 전체 훑어보는 중" }),
    refine: (n, i) => ({ main: n > 1 && i ? `의심 구간 ${i}/${n} 자세히 보는 중` : "의심 구간 자세히 보는 중" }),
    scan: () => ({ main: "영상 분석 중" }),
    events: () => ({ main: "사고 구간 정리 중" }),
    render: (n, i) => ({ main: n > 1 && i ? `사고 클립 ${i}/${n} 만드는 중` : "사고 클립 만드는 중" }),
  };

  // '~하는 중' 뒤의 점: . → .. → ... → .... 를 0.5초마다 반복한다.
  // 점 자리는 고정 폭(.phase-dots)이라 점 개수가 바뀌어도 뒤 글자가 흔들리지 않는다.
  const phaseText = (pct, main, extra) => (
    <>
      {pct != null && `${Math.round(pct)}% · `}
      {main}
      <span className="phase-dots">{".".repeat(1 + (Math.floor(now / 500) % 4))}</span>
      {extra && ` · ${extra}`}
    </>
  );

  const jobProgress = (job) => {
    const key = job.taskId ?? job.videoId;
    const hasReal = typeof job.progress === "number" && (job.progress > 0 || !!job.phase);
    // 단계: 요청(서버 대기열에서 차례를 기다림) → 준비 → 진행(서버가 단계·진행도를 보냄)
    // 다른 사용자의 작업 정보는 드러내지 않으므로 대기 순번은 보여 주지 않는다.
    const stage = hasReal ? "running" : job.status === "PENDING" ? "queued" : "starting";

    // 진행률(막대·%): 서버 값(2초마다, 소수 첫째 자리) 사이를 최근 진행 속도로 채워 계속 조금씩
    // 움직이게 한다. 서버 값보다 최대 3%p(약 4초어치)까지만 앞서가고, 뒤로는 가지 않으며,
    // 서버 값이 크게 뛰어도 0.5초마다 남은 차이의 40%씩 따라가 부드럽게 움직인다.
    const serverP = hasReal ? Math.min(99, job.progress) : 0;
    const sm = progressSmoothRef.current[key]
      || (progressSmoothRef.current[key] = { disp: 0, p: null, at: null, rate: 0, tick: null });
    if (sm.tick !== now) {                        // 같은 순간 여러 곳에서 불러도 한 번만 계산
      const at = job.progressAt ?? now;
      if (sm.p === null || at !== sm.at) {        // 새 서버 값
        if (sm.p !== null && at > sm.at && serverP >= sm.p) {
          const r = (serverP - sm.p) / ((at - sm.at) / 1000);
          sm.rate = sm.rate ? sm.rate * 0.6 + r * 0.4 : r;   // 진행 속도(%/초)도 부드럽게
        }
        sm.p = serverP;
        sm.at = at;
      }
      const ahead = Math.min(3, sm.rate * 4);
      const guess = Math.min(99, sm.p + Math.min(ahead, sm.rate * Math.max(0, (now - sm.at) / 1000)));
      const target = Math.max(sm.disp, guess);
      sm.disp += (target - sm.disp) * 0.4;
      sm.tick = now;
    }
    const pct = hasReal ? sm.disp : 0;

    const ph = PHASE_TEXT[job.phase]?.(job.phaseCount, job.phaseIndex) ?? { main: "분석 중" };
    // 상단 대표 칸: 단계 문구(가운데)와 진행률(맨 오른쪽, pctText)을 따로 보여 준다
    const statusText =
      stage === "queued" ? phaseText(null, "서버에 분석을 요청하는 중")
        : stage === "starting" ? phaseText(null, "분석 준비 중")
          : phaseText(null, ph.main, ph.extra);
    const pctText = stage === "running" ? `${Math.round(pct)}%` : null;
    // 서버 병목으로 기다린다는 걸 알리는 게 목적이라, 요청 단계 문구는 줄이지 않는다
    const shortText =
      stage === "queued" ? "서버에 분석을 요청하는 중" : stage === "starting" ? "준비 중" : `${Math.round(pct)}%`;
    // 펼친 목록의 아랫줄 — 윗줄(짧은 칸: %)과 겹치지 않게 단계만
    const detailText =
      stage === "queued" ? "서버의 앞선 분석이 끝나면 자동으로 시작됩니다"
        : stage === "starting" ? phaseText(null, "분석 준비 중") : phaseText(null, ph.main, ph.extra);
    return { pct, preparing: stage !== "running", stage, statusText, pctText, shortText, detailText };
  };

  // 오버레이 div 안에서 실제 영상 콘텐츠가 차지하는 영역(object-fit:contain 레터박스)과
  // 원본해상도→화면 스케일을 반환. bbox 그리기/클릭 좌표 환산을 이걸로 통일한다.
  // 화면 좌표(포인터 위치)에 있는 차량 탐지 박스 — 없으면 null
  const detectedBoxAt = (e) => {
    const geom = getVideoGeom();
    const el = bboxOverlayRef.current;
    if (!geom || !el || detectedBoxes.length === 0) return null;
    const rect = el.getBoundingClientRect();
    const realX = (e.clientX - rect.left - geom.offsetX) / geom.scale;
    const realY = (e.clientY - rect.top - geom.offsetY) / geom.scale;
    return detectedBoxes.find(({ bbox: [x1, y1, x2, y2] }) => realX >= x1 && realX <= x2 && realY >= y1 && realY <= y2) || null;
  };

  // 영상 칸 크기(w, h) → 원본 해상도와의 배율·레터박스 여백. 해상도는 서버가 분석에 쓰는 크기(DB width/height)
  const geomFor = (w, h) => {
    const vw = selectedVideo?.width || w;
    const vh = selectedVideo?.height || h;
    const scale = Math.min(w / vw, h / vh); // contain
    return {
      scale,
      vw,
      vh,
      offsetX: (w - vw * scale) / 2, // 좌우 레터박스 여백
      offsetY: (h - vh * scale) / 2, // 상하 레터박스 여백
    };
  };
  // 마우스·터치 처리용 — 지금 영상 칸 크기를 직접 잰다
  const getVideoGeom = () => {
    const el = bboxOverlayRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    return geomFor(rect.width, rect.height);
  };
  // 그리기용 — ResizeObserver 가 기록한 영상 칸 크기로 계산(크기가 바뀌면 다시 그려진다)
  const renderGeom = overlaySize ? geomFor(overlaySize.w, overlaySize.h) : null;

  // 고른 차량 박스(bboxList)는 원본 영상 해상도 좌표로 저장한다.
  // 영상 칸 크기가 바뀌어도(창 크기 · 오른쪽 패널 · 영화관 모드 · 전체화면) 그릴 때마다 지금 크기로 바꾸므로
  // 박스가 영상의 같은 자리에 있고, 분석에 보내는 좌표도 그대로다.
  // 화면(오버레이) 좌표 → 원본 좌표 (영상 밖 레터박스로 나간 부분은 영상 안으로 자른다)
  const toOrigBox = (x1, y1, x2, y2) => {
    const g = getVideoGeom();
    if (!g) return null;
    const ox = (x) => Math.round(Math.min(g.vw, Math.max(0, (x - g.offsetX) / g.scale)));
    const oy = (y) => Math.round(Math.min(g.vh, Math.max(0, (y - g.offsetY) / g.scale)));
    return { xmin: ox(x1), ymin: oy(y1), xmax: ox(x2), ymax: oy(y2) };
  };
  // 원본 좌표 → 지금 영상 칸의 화면 위치
  const toDispBox = (box) => {
    const g = renderGeom;
    if (!g) return null;
    return {
      left: g.offsetX + box.xmin * g.scale,
      top: g.offsetY + box.ymin * g.scale,
      width: (box.xmax - box.xmin) * g.scale,
      height: (box.ymax - box.ymin) * g.scale,
    };
  };

  // 넓은 화면의 영상 재생 화면에선 버튼 묶음을 왼쪽 칸 맨 위(사고 클립 사이드바 위)에 둔다.
  // 좁은 화면(1024px 이하)은 한 줄로 쌓여 사이드바가 영상 아래로 가므로 영상 위 헤더에 둔다.
  const isWideScreen = useMediaQuery("(min-width: 1025px)");
  const actionsInSide = isWideScreen && !!selectedVideo;

  // 진행 단계(01 차량 선택 → 02 사고 구간 찾기 → 03 클립 검토).
  // 넓은 화면(1280px 이상)은 영상 오른쪽 칸 위에 세로로, 그보다 좁으면 작업 헤더 안에 가로로 둔다.
  const isXLScreen = useMediaQuery("(min-width: 1280px)");
  // 지금 할 단계 하나를 화면 상태로 정한다: 분석 중 → 02, 차량 고르는 중 → 01, 결과 있음 → 03,
  // 박스만 지정됨 → 02(찾기 차례), 그 외 → 01. 앞 단계는 끝남(✓), 뒤 단계는 아직(흐리게).
  const currentStep = !selectedVideo ? 1
    : isAnalyzing ? 2
      : selectionMode !== "none" ? 1
        : selectedVideo.events.length > 0 ? 3
          : bboxList.length > 0 ? 2
            : 1;
  const selectedVideoJob = selectedVideo ? analyzingJobs.find((j) => j.videoId === selectedVideo.id) : null;
  // 차량 탐지·직접 지정 중 — 영상 크기·비율이 바뀌면 그리던 박스·탐지 박스 위치가 흔들리므로
  // 이 동안엔 영상 크기를 바꾸는 기능(영화관 모드·전체화면·오른쪽 패널 열기/닫기)을 쓰지 못하게 한다
  const isChoosingVehicle = selectionMode !== "none" || isDetectingVehicles;
  const camEnabled = clipView === "cam" && !isChoosingVehicle;
  const camShown = camEnabled && selectedVideo
    ? clipEventsAhead(selectedVideo.events, currentTime, currentEventId, 3) : [];
  const camKey = camShown.map((e) => e.id).join(",");

  // 원본 재생 시각과 CAM 클립을 맞추고 현재 구간에 해당하는 클립만 겹쳐 보여 준다.
  useEffect(() => {
    const main = videoElRef.current;
    const overlay = bboxOverlayRef.current;
    if (!main || !overlay || !camKey) return undefined;
    const events = selectedVideo?.events || [];
    const layers = [...overlay.querySelectorAll(".cam-clip-layer")]
      .map((layer) => ({
        layer,
        cam: layer.querySelector("video"),
        ev: events.find((e) => e.id === Number(layer.dataset.eventId)),
      }))
      .filter((x) => x.cam && x.ev);
    const chip = camChipRef.current;
    const sync = () => {
      const now = clipEventAt(events, main.currentTime, currentEventId);
      let shown = null;
      layers.forEach(({ layer, cam, ev }) => {
        const local = main.currentTime - ev.clipStartSec;
        const inClip = local >= 0 && local < (cam.duration || 0);
        if (inClip) {
          if (Math.abs(cam.currentTime - local) > (main.paused ? 0.01 : 0.12)) cam.currentTime = local;
          if (cam.playbackRate !== main.playbackRate) cam.playbackRate = main.playbackRate;
          if (!main.paused && cam.paused) cam.play().catch(() => {});
          if (main.paused && !cam.paused) cam.pause();
        } else if (!cam.paused) {
          cam.pause();
        }
        const show = now?.id === ev.id && inClip && cam.readyState >= 2;
        layer.classList.toggle("shown", show);
        if (show) shown = ev;
      });
      if (chip) {
        chip.hidden = !shown;
        if (shown) chip.textContent = shown.analysisNo != null ? `분석 #${shown.analysisNo} 충돌 표시` : "충돌 표시";
      }
    };
    let raf = 0;
    const loop = () => { sync(); raf = requestAnimationFrame(loop); };
    const onPlay = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(loop); };
    const onPause = () => { cancelAnimationFrame(raf); sync(); };
    const mainEvents = { play: onPlay, pause: onPause, seeking: sync, seeked: sync, ratechange: sync };
    const camEvents = { loadedmetadata: sync, loadeddata: sync, seeked: sync };
    Object.entries(mainEvents).forEach(([k, fn]) => main.addEventListener(k, fn));
    layers.forEach(({ cam }) => Object.entries(camEvents).forEach(([k, fn]) => cam.addEventListener(k, fn)));
    if (main.paused) sync(); else onPlay();
    return () => {
      cancelAnimationFrame(raf);
      Object.entries(mainEvents).forEach(([k, fn]) => main.removeEventListener(k, fn));
      layers.forEach(({ cam }) => Object.entries(camEvents).forEach(([k, fn]) => cam.removeEventListener(k, fn)));
      if (chip) chip.hidden = true;
    };
  }, [camKey, selectedVideo, currentEventId]);

  // 분석 이력에서 고른 분석의 클립이 아직 남아 있으면 '강조 중' (클립을 다 지우면 저절로 풀림)
  const isHighlighting = !!selectedVideo && highlightAnalysisNo != null
    && selectedVideo.events.some((e) => e.analysisNo === highlightAnalysisNo);
  // 왼쪽 목록은 가장 최근 분석을 먼저, 같은 분석 안에서는 영상 시각 순으로 보인다.
  const listedEvents = selectedVideo
    ? [...selectedVideo.events].sort((a, b) => (b.analysisNo ?? 0) - (a.analysisNo ?? 0) || a.startSec - b.startSec)
    : [];
  const workflowSteps = selectedVideo ? (
    <ol className="incident-workflow" aria-label="사고 의심 구간 검토 순서">
      {[
        { n: 1, title: "분석할 차량 선택", desc: "자동 탐지 또는 직접 지정" },
        {
          n: 2,
          title: "사고 구간 찾기",
          desc: isAnalyzing
            ? `분석 중 · ${typeof selectedVideoJob?.progress === "number" ? `${Math.round(selectedVideoJob.progress)}%` : "준비 중"}`
            : "선택한 차량의 접촉 장면 탐색",
        },
        { n: 3, title: "클립 검토", desc: "원본과 의심 장면을 함께 확인", count: selectedVideo.events.length },
      ].map((st) => {
        const state = st.n < currentStep ? "is-complete" : st.n === currentStep ? "is-current" : "is-pending";
        return (
          <li key={st.n} className={state} aria-current={st.n === currentStep ? "step" : undefined}>
            <span className="step-num" aria-hidden="true">
              {st.n < currentStep ? (
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5" /></svg>
              ) : String(st.n).padStart(2, "0")}
            </span>
            <div><strong>{st.title}</strong><small>{st.desc}</small></div>
            {st.count > 0 && <b>{st.count}건</b>}
          </li>
        );
      })}
    </ol>
  ) : null;

  // 영상 재생 화면의 차량 탐지 · 직접 차량 지정 · 사고감지 실행 버튼 묶음.
  // 차량 탐지 버튼은 상태에 따라 세 가지로 바뀐다:
  // 탐지 전 '차량 탐지' → 탐지 중 '탐지 취소' → 탐지 후 차량을 고르는 동안 '차량 선택 종료'
  const watchActionButtons = selectedVideo ? (
    <div className="watch-header-actions">
      <button
        className={`auto-detect-btn ${selectionMode === "auto" ? "active" : ""} ${isDetectingVehicles ? "detecting" : ""}`}
        disabled={isAnalyzing}
        onClick={isDetectingVehicles ? handleCancelDetectVehicles
          : selectionMode === "auto" ? handleEndDesignateMode : handleAutoDetectVehicles}
        title={isDetectingVehicles ? "진행 중인 차량 탐지를 취소합니다"
          : selectionMode === "auto" ? "차량 고르기를 끝냅니다"
            : "현재 화면의 차량을 자동으로 찾아 표시합니다"}
      >
        {isDetectingVehicles ? "🛑 탐지 취소" : selectionMode === "auto" ? "차량 선택 종료" : "차량 탐지"}
      </button>

      {/* 직접 차량 지정 + 지정 모드 안내 말풍선(버튼 위로 떠서 상단 바의 빈 곳에 겹쳐 뜬다 — 아무것도 밀어내지 않음) */}
      <div className="designate-btn-wrapper">
        <button
          className={`bbox-designate-btn ${selectionMode === "manual" ? "active" : ""}`}
          disabled={isAnalyzing || isDetectingVehicles} /* 분석 중 · 차량 자동 탐지 중엔 누를 수 없다 */
          onClick={() => {
            if (selectionMode === "manual") {
              // '직접 지정 종료'는 모드만 끝낸다 — 그린 박스는 남는다(차량 탐지로 고른 박스와 같게).
              // 지우려면 박스에 마우스를 올려 ✕
              handleEndDesignateMode();
            } else {
              handleStartManualDesignate();
            }
          }}
          title={isAnalyzing ? "분석 중에는 지정할 수 없습니다"
            : isDetectingVehicles ? "차량 탐지 중에는 지정할 수 없습니다"
              : (selectionMode === "manual" ? "직접 지정 모드를 끝냅니다" : "마우스로 드래그해 차량을 직접 지정합니다")}
        >
          {selectionMode === "manual" ? "직접 지정 종료" : "직접 차량 지정"}
        </button>
        {selectionMode === "manual" && (
          <div className="designate-hint" role="status">
            🖱️ 마우스로 드래그하여 사고 차량을 지정해 주세요.
          </div>
        )}
      </div>


      {/* 사고감지 실행 버튼 + 확인 팝오버 / 취소 버튼 */}
      <div className="detect-btn-wrapper">
        {isAnalyzing ? (
          <button
            className="detect-run-btn analyzing-cancel-btn"
            onClick={() => {
              const currentJob = analyzingJobs.find((j) => j.videoId === selectedVideo?.id);
              handleCancelAnalysis(currentJob?.taskId);
            }}
            title="진행 중인 사고 감지 분석을 취소합니다"
          >
            🛑 사고감지 취소
          </button>
        ) : (
          <button
            className="detect-run-btn"
            onClick={handleAnalyzeClick}
            title="선택한 차량의 사고 의심 구간을 찾습니다"
          >
            사고 구간 찾기
          </button>
        )}

        {showAnalyzeConfirm && (
          <>
            <div
              className="analyze-popover-overlay"
              onClick={() => setShowAnalyzeConfirm(false)}
            />
            <div className="analyze-popover">
              <p className="analyze-popover-text">
                {selectedVideo?.analyzedBefore
                  ? "이미 분석을 완료한 영상입니다. 다시 분석을 실행하시겠습니까?"
                  : "현재 선택하신 차량의 사고예상 구간을 탐지하시겠습니까?"}
              </p>
              <div className="analyze-popover-actions">
                <button className="analyze-popover-run" onClick={runAnalysis}>
                  실행
                </button>
                <button
                  className="analyze-popover-cancel"
                  onClick={() => setShowAnalyzeConfirm(false)}
                >
                  뒤로가기
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  ) : null;

  return (
    <div className="dashboard-layout">
      {/* 상단 네비게이션 바 */}
      <header className="top-navbar">
        <div className="nav-section nav-left">
          <button className="nav-logo nav-home-btn" onClick={handleBackToHome} aria-label="SIOT 홈으로 이동">
            <span className="workspace-wordmark" translate="no">SI<span className="workspace-lens">O</span>T<span className="workspace-brand-dot">.</span></span><span className="workspace-brand-caption" translate="no">{selectedVideo ? "INCIDENT REVIEW" : "LIBRARY"}</span>
          </button>
        </div>

        <div className="nav-section nav-center">
          {analyzingJobs.length === 0 ? (
            <div className="analysis-status-box analysis-status-idle">
              <span className="analysis-idle-text">현재 분석 중인 영상이 없습니다</span>
            </div>
          ) : (() => {
            const lead = analyzingJobs[0];
            const { pct, preparing, statusText, pctText } = jobProgress(lead);
            return (
              <div className="analysis-status-box">
                <div className="analysis-status-main">
                  <div className="analysis-status-info">
                    <span className="analysis-status-label">
                      {/* 영상 목록의 제목과 같게 (제목을 넣었으면 그 제목, 아니면 '녹화일 녹화본 #N') */}
                      🔍 {(() => {
                        const v = videos.find((x) => x.id === lead.videoId);
                        return v ? getVideoDisplayTitle(v, videos) : lead.dateLabel;
                      })()} 분석 중
                    </span>
                    <span className="analysis-status-eta">
                      {statusText}
                    </span>
                    {pctText && <span className="analysis-status-pct">{pctText}</span>}
                  </div>
                  <div className="analysis-progress-track">
                    <div
                      className={`analysis-progress-fill${preparing ? " is-indeterminate" : ""}`}
                      style={preparing ? undefined : { width: `${pct}%` }}
                    />
                  </div>
                </div>

                <button
                  className="analysis-jobs-toggle"
                  onClick={() => setShowJobsDropdown((v) => !v)}
                  title="분석 중인 영상 목록을 펼칩니다"
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
                                {(() => {
                                  // 같은 날짜 영상이 여럿이면 '녹화본 #N' 으로 구분해 보여준다
                                  const v = videos.find((x) => x.id === job.videoId);
                                  return v ? getVideoDisplayTitle(v, videos) : job.dateLabel;
                                })()}
                              </span>
                              <div className="analysis-job-top-right">
                                <span className="analysis-job-pct">
                                  {p.shortText}
                                </span>
                                <button
                                  className="analysis-job-cancel-btn"
                                  onClick={() => handleCancelAnalysis(job.taskId)}
                                  title="이 분석을 취소합니다"
                                >
                                  🛑 취소
                                </button>
                              </div>
                            </div>
                            <div className="analysis-progress-track">
                              <div
                                className={`analysis-progress-fill${p.preparing ? " is-indeterminate" : ""}`}
                                style={p.preparing ? undefined : { width: `${p.pct}%` }}
                              />
                            </div>
                            <span className="analysis-job-eta">
                              {p.detailText}
                            </span>
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
              aria-expanded={isProfileOpen}
              aria-controls="profile-menu"
            >
              <div className="profile-avatar">
                <span>{avatarInitial}</span>
              </div>
              <span className="profile-trigger-copy"><strong>{displayName}</strong><small>내 계정</small></span><span className="profile-trigger-chevron" aria-hidden="true">⌄</span>
            </button>

            {isProfileOpen && (
              <>
                <div className="dropdown-overlay" onClick={() => setIsProfileOpen(false)} />
                <div className="profile-dropdown-menu" id="profile-menu">
                  {/* 관리자/사용자 정보 요약 Header */}
                  <div className="dropdown-header">
                    <div className="header-avatar">{avatarInitial}</div>
                    <div className="header-info">
                      <span className="info-name">{displayName} ({currentUsername})</span>
                      <span className="info-role">{roleLabel}</span>
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
                    <InterfaceIcon name="user" /> {currentUser?.role === "ADMIN" ? "관리자 정보 수정" : "내 정보 수정"}
                  </button>
                  <button
                    className="dropdown-item"
                    onClick={() => {
                      setIsProfileOpen(false);
                      setIsSettingsOpen(true);
                      setActiveSettingsTab("security");
                    }}
                  >
                    <InterfaceIcon name="lock" /> 비밀번호 변경
                  </button>

                  <div className="dropdown-divider" />

                  {/* 테마 설정 (Appearance) */}
                  <div className="dropdown-section-title">테마 설정</div>
                  <div className="dropdown-item-toggle">
                    <span className="theme-label"><InterfaceIcon name="moon" /> 다크 모드</span>
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
                    <InterfaceIcon name="exit" /> 로그아웃
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </header>

      {/* 컨텐츠 영역: 영상 목록 또는 재생 화면 */}
      {!selectedVideo ? (
        <main className="home-view">
          <section className="incident-overview">
            <div className="incident-overview-copy"><span className="incident-eyebrow" translate="no">SIOT / INCIDENT REVIEW</span><h1>주차된 순간부터,<br />놓치지 않는 사고 단서.</h1><p>CCTV 속 접촉 의심 장면을 찾고, 필요한 구간을 빠르게 검토하세요.</p><button className="incident-upload" onClick={() => setShowUpload(true)}>영상 추가하고 분석 시작 <span aria-hidden="true">↗</span></button></div>
            <div className="incident-metrics" aria-label="내 영상 현황">
              <div><span>보관 영상</span><strong>{videos.length}<small>개</small></strong></div>
              <div><span>사고 의심 구간</span><strong>{videos.reduce((sum, video) => sum + video.events.length, 0)}<small>건</small></strong></div>
              <div><span>분석 진행</span><strong>{analyzingJobs.length}<small>개</small></strong></div>
            </div>
          </section>
          <div className="incident-section-title"><div><span className="incident-eyebrow" translate="no">VIDEO LIBRARY</span><h2>영상 보관함</h2></div></div>
          {/* 기간 필터 및 영상 관리 툴바 */}
          <div className="home-toolbar">
            <div className="home-toolbar-filters">
              <div className="filter-pills">
                <button className={filterDays === 9999 ? "active" : ""} onClick={() => setFilterDays(9999)}>전체</button>
                <button className={filterDays === 7 ? "active" : ""} onClick={() => setFilterDays(7)}>1주일</button>
                <button className={filterDays === 14 ? "active" : ""} onClick={() => setFilterDays(14)}>2주일</button>
                <button className={filterDays === 30 ? "active" : ""} onClick={() => setFilterDays(30)}>1개월</button>
                <button className={filterDays === 90 ? "active" : ""} onClick={() => setFilterDays(90)}>3개월</button>
              </div>

              <div className="list-search-bar">
                <span className="search-icon"><SearchIcon /></span>
                <input
                  type="text"
                  placeholder="영상 날짜, 제목 검색"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  aria-label="영상 검색"
                />
                {searchQuery && (
                  <button className="search-clear-btn" onClick={() => setSearchQuery("")} aria-label="검색어 지우기">
                    ✕
                  </button>
                )}
              </div>
            </div>

            <div className="home-toolbar-actions">
              {deleteMode ? (
                <>
                  <button
                    className="delete-select-all-btn"
                    onClick={() => {
                      if (selectedForDelete.length === filteredVideos.length && filteredVideos.length > 0) {
                        setSelectedForDelete([]);
                      } else {
                        setSelectedForDelete(filteredVideos.map((v) => v.id));
                      }
                    }}
                  >
                    {selectedForDelete.length === filteredVideos.length && filteredVideos.length > 0
                      ? "☑ 전체 해제"
                      : "☐ 전체 선택"}
                  </button>
                  <button className="delete-cancel-btn" onClick={exitDeleteMode}>
                    취소
                  </button>
                  <button className="delete-confirm-btn" onClick={handleDeleteSelected}>
                    🗑 선택 삭제 ({selectedForDelete.length})
                  </button>
                </>
              ) : (
                <>
                  <button className="toolbar-upload-btn" onClick={() => setShowUpload(true)}>
                    ⬆ 영상 업로드
                  </button>
                  <button className="toolbar-delete-btn" onClick={() => setDeleteMode(true)}>
                    🗑 영상 삭제
                  </button>
                </>
              )}
            </div>
          </div>


          {/* 영상 썸네일 그리드 — 열 수는 CSS 가 화면 폭에 맞춰 정하고, 한 페이지 개수가 그 열 수를 따른다 */}
          <div className="video-grid" ref={setVideoGridEl}>
            {filteredVideos.length === 0 ? (
              <div className="empty-state">
                {videoLoadError ? (
                  <>
                    <p>{videoLoadError}</p>
                    <button type="button" onClick={loadVideos} disabled={isLoadingVideos}>
                      {isLoadingVideos ? "영상 목록을 불러오는 중…" : "다시 시도"}
                    </button>
                  </>
                ) : (
                  "선택한 기간에 해당하는 영상이 없습니다."
                )}
              </div>
            ) : (
              paginatedVideos.map((video) => {
                const isSelected = selectedForDelete.includes(video.id);
                return (
                  <div
                    key={video.id}
                    className={`video-card ${deleteMode ? "delete-mode" : ""} ${isSelected ? "delete-selected" : ""}`}
                    onClick={() =>
                      deleteMode ? toggleDeleteSelect(video.id) : handleWatchVideo(video)
                    }
                  >
                    <div className="video-thumbnail">
                      {/* 영상 첫 프레임 썸네일 */}
                      <img
                        className="video-thumbnail-img"
                        src={api.thumbnailUrl(video.id)}
                        alt={`${video.date} 썸네일`}
                        loading="lazy"
                        onError={(e) => { e.currentTarget.style.display = "none"; }}
                      />
                      {/* 왼쪽 아래 녹화일자(제목을 따로 넣으면 언제 찍었는지 모르니) · 오른쪽 아래 의심 구간 수 */}
                      {video.date && <span className="recording-date-badge">{video.date}</span>}
                      <span className="event-count-badge">의심 구간 {video.events.length}건</span>
                      {deleteMode && (
                        <span className={`delete-check ${isSelected ? "checked" : ""}`}>
                          {isSelected ? "✓" : ""}
                        </span>
                      )}
                    </div>
                    <div className="video-info">
                      <div className="video-info-text">
                        <h3>{getVideoDisplayTitle(video, videos)}</h3>
                        <p>영상 길이: {formatTime(video.duration)}</p>
                        {/* 보관 기간 — 지나면 영상과 분석 클립이 함께 지워진다. 7일 이하는 강조 */}
                        {video.expiresInDays != null && (
                          <p
                            className={`video-expiry ${video.expiresInDays <= 7 ? "soon" : ""}`}
                            title="보관 기간이 지나면 영상과 분석된 클립이 함께 삭제됩니다"
                          >
                            {video.expiresInDays > 0 ? `${video.expiresInDays}일 뒤 만료` : "곧 만료"}
                          </p>
                        )}
                      </div>
                      {/* ··· 메뉴: 영상 정보 확인(giwon 정보 팝업 — 업로드 일시 등 포함) / 영상 삭제 */}
                      {!deleteMode && <VideoCardActions video={video} title={getVideoDisplayTitle(video, videos)} onShowInfo={() => setInfoVideo(video)} onDeleted={(id) => {
                        setVideos((previous) => previous.filter((video) => video.id !== id));
                        setSelectedForDelete((previous) => previous.filter((videoId) => videoId !== id));
                        showToast("영상을 삭제했습니다.", "success");
                      }} />}
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {/* 페이지네이션 (영상 개수가 6개를 초과할 때 노출) */}
          {filteredVideos.length > VIDEOS_PER_PAGE && (
            <div className="pagination-bar">
              {/* 버튼 묶음은 화면 가운데, 페이지 정보는 그 오른쪽 — 정보 글자 때문에 버튼이 왼쪽으로 밀리지 않게 */}
              <div className="pagination-controls">
                <button
                  className="pagination-btn pagination-nav"
                  onClick={() => handlePageChange(1)}
                  disabled={currentPage === 1}
                  title="첫 페이지로 이동"
                >
                  «
                </button>
                <button
                  className="pagination-btn pagination-nav"
                  onClick={() => handlePageChange(Math.max(1, currentPage - 1))}
                  disabled={currentPage === 1}
                  title="이전 페이지로 이동"
                >
                  ‹
                </button>

                <div className="pagination-pages">
                  {getPageNumbers().map((pageNum) => (
                    <button
                      key={pageNum}
                      className={`pagination-btn pagination-num ${currentPage === pageNum ? "active" : ""}`}
                      onClick={() => handlePageChange(pageNum)}
                    >
                      {pageNum}
                    </button>
                  ))}
                </div>

                <button
                  className="pagination-btn pagination-nav"
                  onClick={() => handlePageChange(Math.min(totalPages, currentPage + 1))}
                  disabled={currentPage === totalPages}
                  title="다음 페이지로 이동"
                >
                  ›
                </button>
                <button
                  className="pagination-btn pagination-nav"
                  onClick={() => handlePageChange(totalPages)}
                  disabled={currentPage === totalPages}
                  title="마지막 페이지로 이동"
                >
                  »
                </button>
              </div>

              <span className="pagination-info">
                {currentPage} / {totalPages} 페이지 (총 {filteredVideos.length}개)
              </span>
            </div>
          )}
        </main>
      ) : (
        <main className={`watch-view ${isTheaterMode ? "theater-view" : ""}`}>
          {!isTheaterMode && <section className="incident-workspace-header">
            <div className="incident-workspace-top"><div><button className="incident-back" onClick={handleBackToHome}>← 영상 보관함</button><h1>{getVideoDisplayTitle(selectedVideo, videos)}</h1></div>
              {/* 오른쪽 패널(진행 단계·분석 이력) 열기·닫기 — 닫으면 그 칸이 빠지고 영상이 넓어진다(넓은 화면만).
                  영상 크기가 바뀌므로 차량을 고르는 중엔 막는다 */}
              {isXLScreen && (
                <button
                  className="panel-toggle-btn"
                  onClick={() => setIsHistoryOpen((open) => !open)}
                  disabled={isChoosingVehicle}
                  aria-expanded={isHistoryOpen}
                  title={isChoosingVehicle ? "차량을 고르는 중에는 쓸 수 없습니다"
                    : isHistoryOpen ? "진행 단계·분석 이력 패널을 닫고 영상을 넓게 봅니다" : "진행 단계·분석 이력 패널을 엽니다"}
                >
                  {/* 오른쪽 패널 모양 아이콘 — 열려 있으면 오른쪽 칸이 칠해지고, 닫혀 있으면 빈 칸 */}
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" aria-hidden="true">
                    <rect x="3" y="4.5" width="18" height="15" rx="1.5" />
                    <path d="M15 4.5v15" />
                    {isHistoryOpen && <rect x="15" y="4.5" width="6" height="15" fill="currentColor" stroke="none" opacity="0.45" />}
                  </svg>
                  {isHistoryOpen ? "오른쪽 패널 닫기" : "오른쪽 패널 열기"}
                </button>
              )}
            </div>
            {!isXLScreen && workflowSteps}
          </section>}

          {/* 좁은 화면(1024px 이하): 영상 위 헤더에 버튼 묶음.
              넓은 화면은 헤더 줄 없이 영상을 맨 위로 — 버튼은 왼쪽 칸 맨 위에 둔다 */}
          {!actionsInSide && (
            <div className="watch-header">
              <div className="watch-header-right">
                {watchActionButtons}
              </div>
            </div>
          )}

          <div className={`watch-layout ${isTheaterMode ? "theater-mode" : ""} ${!isHistoryOpen && isXLScreen ? "history-collapsed" : ""}`}>
            {/* 왼쪽 칸: (넓은 화면) 버튼 묶음 + 사고 클립 사이드바 */}
            <div className="watch-side">
              {actionsInSide && watchActionButtons}
              <aside className={`event-sidebar ${isHighlighting ? "highlighting" : ""}`}>
                {/* 머리말: 윗줄 제목 + 버튼(가운데 맞춤), 아랫줄 건수 안내(가로 전체 — 버튼 때문에 줄바꿈되지 않게) */}
                <div className="event-sidebar-header">
                  <h3>사고 의심 구간</h3>
                  {/* 전체 삭제 (목록 닫기(×)는 사용자 요청으로 뺐다, 2026-10-27) */}
                  <div className="event-header-actions">
                    {selectedVideo.events.length > 0 && (
                      <button
                        className="event-clear-all-btn"
                        onClick={requestClearAllEvents}
                        title="이 영상의 사고 의심 클립을 모두 삭제합니다"
                      >
                        🗑 전체 삭제
                      </button>
                    )}
                  </div>
                  <p className="event-summary">클립 {selectedVideo.events.length}건 · 직접 검토가 필요합니다.</p>
                </div>

                <div className="event-list" ref={eventListRef}>
                  {selectedVideo.events.length === 0 ? (
                    <div className="event-empty-hint">감지된 이벤트가 없습니다.</div>
                  ) : (
                    listedEvents.map((event) => (
                      <div
                        key={event.id}
                        className={`event-item ${currentEventId === event.id ? "active" : ""} ${
                          !isHighlighting
                            ? ""
                            : event.analysisNo === highlightAnalysisNo ? "analysis-highlight" : "analysis-dim"
                        }`}
                        onClick={() => seekToEvent(event)}
                      >
                        <div className="event-item-top">
                          <span className="event-time-pill">{formatTime(event.timestamp)}</span>
                          <span className="event-actual-time-text" title="이 영상의 몇 번째 분석에서 나온 클립인지 나타냅니다">
                            {event.analysisNo != null && `분석 #${event.analysisNo}`}
                          </span>
                          <div className="event-item-actions">
                            <button
                              className="event-play-icon-btn"
                              onClick={(e) => {
                                e.stopPropagation();
                                setCurrentEventId(event.id);
                                if (event.hasClip) {
                                  setClipEvent(event);
                                } else {
                                  showToast("이 이벤트의 사고 클립이 아직 없습니다.", "warning");
                                }
                              }}
                              aria-label={`${formatTime(event.timestamp)} 사고구간 클립 재생`}
                              title="사고 구간 클립을 봅니다"
                            >
                              {/* 글자 ▶ 는 글꼴마다 모양이 다르고 macOS·iOS 에선 이모지(▶️)로 바뀌어 SVG 로 그린다 */}
                              <PlayerIcon name="play" size={14} />
                            </button>
                            <button
                              className="event-delete-item-btn"
                              onClick={(e) => requestDeleteEvent(event, e)}
                              title="이 클립을 삭제합니다"
                              aria-label={`${formatTime(event.timestamp)} 사고구간 클립 삭제`}
                            >
                              <TrashIcon size={15} />
                            </button>
                          </div>
                        </div>
                        <div className="event-item-bottom">
                          <h4 className="event-title-text" title="분석 일시">
                            {(() => {
                              const at = formatAnalyzedAt(event.analyzedAt);
                              return at && (
                                <>
                                  {at[0]}
                                  <span className="event-analyzed-time">{at[1]}</span>
                                </>
                              );
                            })()}
                          </h4>
                        </div>
                      </div>
                    ))
                  )}
                </div>


                {/* 달력을 사이드바 이벤트 목록 하단으로 삽입 */}
                <div className="sidebar-calendar-container">
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
                            title={hasVideo ? `총 ${videos.length}개 영상 / ${totalEvents}건의 이벤트` : "영상 없음"}
                          >
                            <span className="calendar-date-number">{date.getDate()}</span>
                            {videos.length > 1 && (
                              <span className="calendar-multi-count-badge" title={`영상 ${videos.length}개`}>
                                {videos.length}개
                              </span>
                            )}
                          </button>
                        );
                      })}
                    </div>
                  </section>
                  <div className="calendar-legend"><span><i /> 영상 있음</span><span><i /> 의심 구간 있음</span><span><i /> 선택한 날짜</span></div>
                </div>
              </aside>
            </div>

            {/* 우측/중앙: 메인 비디오 플레이어 */}
            <section className="player-section">
              {/* 마지막 분석이 실패·중단됐으면 같은 차량 박스로 다시 실행하도록 안내 */}
              {!isAnalyzing && selectedVideo?.lastTask?.status === "FAILURE" && (
                <div
                  className="analysis-failed-banner"
                  role="status"
                  title={selectedVideo.lastTask.errorMessage || ""}
                >
                  <span>
                    {(selectedVideo.lastTask.errorMessage || "").startsWith("분석 서버가 재시작")
                      ? "⚠️ 서버 재시작으로 분석이 중단됐습니다"
                      : "⚠️ 마지막 분석이 실패했습니다"}
                  </span>
                  <button className="analysis-retry-btn" onClick={handleRetryAnalysis}>
                    다시 실행
                  </button>
                </div>
              )}

              <div className="player-container" ref={playerContainerRef}>
                <div
                  className={`mock-video-player ${isBBoxMode ? "bbox-mode-active" : ""}`}
                  ref={bboxOverlayRef}
                  style={{
                    position: "relative",
                    userSelect: "none",
                    aspectRatio: selectedVideo?.width && selectedVideo?.height
                      ? `${selectedVideo.width} / ${selectedVideo.height}`
                      : "16 / 9",
                  }}
                  /* 박스 그리기·차량 고르기는 pointer 이벤트 — 마우스·터치(폰·태블릿)·펜 모두 같은 코드로 동작 */
                  onPointerDown={(e) => {
                    if (!isBBoxMode) return;
                    if (e.target.closest("button")) return; // 박스 ✕ 등 버튼을 누를 땐 그리기를 시작하지 않는다
                    // 터치에는 hover 가 없으니 누른 위치의 박스를 바로 찾는다
                    const picked = selectionMode === "auto" ? hoveredDetectedBox || detectedBoxAt(e) : null;
                    if (selectionMode === "auto" && picked) {
                      const [x1, y1, x2, y2] = picked.bbox; // 탐지 결과는 이미 원본 해상도 좌표
                      setBboxList([{ id: ++bboxSeq, xmin: x1, ymin: y1, xmax: x2, ymax: y2 }]);
                      showToast("차량이 선택되었습니다.", "success");
                      return;
                    }
                    if (selectionMode === "manual") {
                      e.preventDefault();
                      // 손가락·마우스가 영상 밖으로 나가도 끝까지 따라가게 포인터를 붙잡는다
                      e.currentTarget.setPointerCapture?.(e.pointerId);
                      const rect = bboxOverlayRef.current.getBoundingClientRect();
                      const x = Math.round(e.clientX - rect.left);
                      const y = Math.round(e.clientY - rect.top);
                      setIsDrawing(true);
                      setCurrentDraw({ startX: x, startY: y, endX: x, endY: y });
                    }
                  }}
                  onPointerMove={(e) => {
                    if (!isBBoxMode) return;
                    const rect = bboxOverlayRef.current.getBoundingClientRect();

                    if (selectionMode === "manual" && isDrawing) {
                      const x = Math.round(e.clientX - rect.left);
                      const y = Math.round(e.clientY - rect.top);
                      setCurrentDraw((prev) => (prev ? { ...prev, endX: x, endY: y } : null));
                      return;
                    }

                    if (selectionMode === "auto" && detectedBoxes.length > 0 && getVideoGeom()) {
                      const { scale, offsetX, offsetY } = getVideoGeom();
                      // 화면(오버레이) 좌표 → 레터박스 여백 제거 → 원본 해상도 좌표
                      const realX = (e.clientX - rect.left - offsetX) / scale;
                      const realY = (e.clientY - rect.top - offsetY) / scale;

                      const hit = detectedBoxes.find((box) => {
                        const [x1, y1, x2, y2] = box.bbox;
                        return realX >= x1 && realX <= x2 && realY >= y1 && realY <= y2;
                      });

                      setHoveredDetectedBox(hit || null);
                    } else if (selectionMode !== "auto") {
                      setHoveredDetectedBox(null);
                    }
                  }}
                  onPointerUp={(e) => {
                    if (e.currentTarget.hasPointerCapture?.(e.pointerId)) {
                      e.currentTarget.releasePointerCapture(e.pointerId);
                    }
                    if (selectionMode === "manual" && isDrawing && currentDraw) {
                      const rect = bboxOverlayRef.current.getBoundingClientRect();
                      const x = Math.round(e.clientX - rect.left);
                      const y = Math.round(e.clientY - rect.top);
                      const xmin = Math.min(currentDraw.startX, x);
                      const ymin = Math.min(currentDraw.startY, y);
                      const xmax = Math.max(currentDraw.startX, x);
                      const ymax = Math.max(currentDraw.startY, y);
                      const orig = xmax - xmin > 5 && ymax - ymin > 5 ? toOrigBox(xmin, ymin, xmax, ymax) : null;
                      if (orig) setBboxList([{ id: ++bboxSeq, ...orig }]);
                      setCurrentDraw(null);
                      setIsDrawing(false);
                    }
                  }}
                  onPointerCancel={() => {
                    // 시스템 제스처 등으로 끊기면 그리던 박스는 버린다
                    setCurrentDraw(null);
                    setIsDrawing(false);
                  }}
                  onPointerLeave={() => {
                    if (selectionMode === "manual" && isDrawing && currentDraw) {
                      const xmin = Math.min(currentDraw.startX, currentDraw.endX);
                      const ymin = Math.min(currentDraw.startY, currentDraw.endY);
                      const xmax = Math.max(currentDraw.startX, currentDraw.endX);
                      const ymax = Math.max(currentDraw.startY, currentDraw.endY);
                      const orig = xmax - xmin > 5 && ymax - ymin > 5 ? toOrigBox(xmin, ymin, xmax, ymax) : null;
                      if (orig) setBboxList([{ id: ++bboxSeq, ...orig }]);
                    }
                    setCurrentDraw(null);
                    setIsDrawing(false);
                    setHoveredDetectedBox(null);
                  }}
                >

                  {/* 실제 업로드 영상 재생 (bbox 모드에서는 포인터 이벤트를 컨테이너로 전달) */}
                  <video
                    ref={videoElRef}
                    className="real-video"
                    src={api.streamUrl(selectedVideo.id)}
                    poster={api.thumbnailUrl(selectedVideo.id)}
                    preload="auto"
                    playsInline
                    controls={false}

                    onPlay={(e) => {
                      // 키보드 미디어 키·브라우저/OS 미디어 조작으로 재생돼도 탐지 중이면 바로 멈춘다
                      if (isDetectingVehicles) {
                        e.currentTarget.pause();
                        return;
                      }
                      // 탐지 박스는 탐지한 순간의 화면 기준이라 재생하면 어긋난다 → 자동 탐지 모드를 끝낸다(고른 박스는 남는다)
                      if (selectionMode === "auto") handleEndDesignateMode();
                      setIsPlaying(true);
                    }}
                    onSeeking={() => {
                      // 다른 장면으로 옮겨도(재생바·사고 의심 구간) 마찬가지
                      if (selectionMode === "auto" && !isDetectingVehicles) handleEndDesignateMode();
                    }}
                    onPause={() => setIsPlaying(false)}
                    onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
                    onLoadedMetadata={(e) => {
                      const d = e.currentTarget.duration;
                      setMediaDuration(Number.isFinite(d) ? d : 0);
                    }}
                    onDurationChange={(e) => {
                      const d = e.currentTarget.duration;
                      setMediaDuration(Number.isFinite(d) ? d : 0);
                    }}
                    style={{
                      display: "block",
                      width: "100%",
                      height: "100%",
                      objectFit: "contain",
                      background: "#000",
                      pointerEvents: isBBoxMode ? "none" : "auto",
                    }}
                  />

                  {/* CAM 클립을 원본 영상 시각에 맞춰 영상 그림 영역 위에 겹친다. */}
                  {camShown.map((ev) => (
                    <div
                      key={ev.id}
                      data-event-id={ev.id}
                      className="cam-clip-layer"
                      style={renderGeom ? {
                        left: renderGeom.offsetX,
                        top: renderGeom.offsetY,
                        width: renderGeom.vw * renderGeom.scale,
                        height: renderGeom.vh * renderGeom.scale,
                      } : { inset: 0 }}
                      aria-hidden="true"
                    >
                      <video src={api.clipUrl(ev.id)} muted playsInline preload="auto" />
                    </div>
                  ))}

                  {/* 차량 자동 탐지 중: 영상 화면 위에 로딩 표시 */}
                  {isDetectingVehicles && (
                    <div className="vehicle-detect-loading" role="status" aria-live="polite">
                      <div className="vehicle-detect-spinner" />
                      <span>차량을 찾는 중…</span>
                    </div>
                  )}

                  {/* 저장된 바운딩 박스 렌더링 (단일 박스) */}
                  {bboxList.map((box) => {
                    const d = toDispBox(box); // 원본 좌표 → 지금 영상 칸 위치(크기가 바뀌면 다시 그려 따라간다)
                    return d && (
                    <div
                      key={box.id}
                      className={`bbox-drawn ${isBBoxMode ? "editing" : ""}`}
                      style={{ left: d.left, top: d.top, width: d.width, height: d.height }}
                    >
                      {/* 삭제(✕): 직접 지정 중엔 늘 보이고, 그 밖엔(차량 탐지로 고른 뒤 모드를 끝내도) 박스에 마우스를 올리면 보인다.
                          분석 중엔 숨긴다 — 분석은 시작할 때의 박스로 돌기 때문 */}
                      {!isAnalyzing && (
                        <button
                          className="bbox-delete-btn"
                          onClick={(e) => {
                            e.stopPropagation();
                            setBboxList((prev) => prev.filter((b) => b.id !== box.id));
                          }}
                          title="지정한 박스를 삭제합니다"
                        >
                          ✕
                        </button>
                      )}
                    </div>
                    );
                  })}

                  {/* 현재 수동 드래그 중인 박스 미리보기 */}
                  {selectionMode === "manual" && isDrawing && currentDraw && (() => {
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

                  {/* YOLO 자동 탐지 모드: 탐지된 모든 bbox를 항상 표시 (진단용 #번호·클래스·신뢰도) */}
                  {selectionMode === "auto" && detectedBoxes.length > 0 && renderGeom && (() => {
                    const { scale, offsetX, offsetY } = renderGeom;
                    return detectedBoxes.map((box) => {
                      const [x1, y1, x2, y2] = box.bbox;
                      const dispX1 = Math.round(offsetX + x1 * scale);
                      const dispY1 = Math.round(offsetY + y1 * scale);
                      const dispX2 = Math.round(offsetX + x2 * scale);
                      const dispY2 = Math.round(offsetY + y2 * scale);
                      const isHovered = hoveredDetectedBox && hoveredDetectedBox.id === box.id;
                      return (
                        <div
                          key={box.id}
                          style={{
                            position: "absolute",
                            left: dispX1,
                            top: dispY1,
                            width: Math.max(4, dispX2 - dispX1),
                            height: Math.max(4, dispY2 - dispY1),
                            border: `2px solid ${isHovered ? "#16a34a" : "rgba(34,197,94,0.7)"}`,
                            boxSizing: "border-box",
                            borderRadius: 2,
                            pointerEvents: "none",
                          }}
                        >
                        </div>
                      );
                    });
                  })()}

                  {/* YOLO 자동 탐지 모드: 마우스 Hover 미리보기 (Hover 시에만 가상 노출) */}
                  {selectionMode === "auto" && hoveredDetectedBox && renderGeom && (() => {
                    const { scale, offsetX, offsetY } = renderGeom;
                    const [x1, y1, x2, y2] = hoveredDetectedBox.bbox;
                    const dispX1 = Math.round(offsetX + x1 * scale);
                    const dispY1 = Math.round(offsetY + y1 * scale);
                    const dispX2 = Math.round(offsetX + x2 * scale);
                    const dispY2 = Math.round(offsetY + y2 * scale);

                    return (
                      <div
                        className="bbox-hover-preview"
                        style={{
                          left: dispX1,
                          top: dispY1,
                          width: Math.max(10, dispX2 - dispX1),
                          height: Math.max(10, dispY2 - dispY1),
                        }}
                      >
                        <span className="bbox-hover-label">
                          🚗 클릭하여 이 차량 선택
                        </span>
                      </div>
                    );
                  })()}
                </div>

                {/* 재생바 + 재생 컨트롤 (영상 화면을 가리지 않게 화면 바깥 하단) */}
                <div className="custom-progress-bar">
                  <div
                    className={`progress-track ${isSeeking ? "seeking" : ""} ${isDetectingVehicles ? "locked" : ""}`}
                    title={isDetectingVehicles ? "차량 탐지 중에는 이동할 수 없습니다" : undefined}
                    onPointerDown={handleSeekStart}
                    onPointerMove={handleSeekMove}
                    onPointerUp={handleSeekEnd}
                    onPointerCancel={handleSeekEnd}
                  >
                    <div className="progress-rail">
                      {/* 실제 재생 진행률 */}
                      <div
                        className="progress-fill"
                        style={{ width: `${playerDuration ? Math.min(100, (currentTime / playerDuration) * 100) : 0}%` }}
                      />

                      {/* 사고 의심 구간 (빨간 구간) — 누르면 그 시점부터 재생 */}
                      {playerDuration > 0 &&
                        selectedVideo.events.map((event) => {
                          const start = event.startSec ?? event.timestamp;
                          const end = Math.max(start, event.endSec ?? event.endTimestamp ?? start);
                          const left = Math.min(100, (start / playerDuration) * 100);
                          const width = Math.min(100 - left, ((end - start) / playerDuration) * 100);
                          return (
                            <div
                              key={event.id}
                              className={`event-range ${currentEventId === event.id ? "active" : ""}`}
                              style={{ left: `${left}%`, width: `${width}%` }}
                              title={`${event.title} ${formatTime(Math.floor(start))} ~ ${formatTime(Math.ceil(end))}`}
                              onPointerDown={(e) => e.stopPropagation()}
                              onClick={(e) => {
                                e.stopPropagation();
                                seekToEvent(event);
                              }}
                            />
                          );
                        })}
                    </div>
                    <div
                      className="progress-thumb"
                      style={{ left: `${playerDuration ? Math.min(100, (currentTime / playerDuration) * 100) : 0}%` }}
                    />
                    <div className="progress-hover-time" ref={seekHoverRef} />
                  </div>

                  <div className="player-controls">
                    <button
                      className="control-btn"
                      onClick={togglePlay}
                      disabled={isDetectingVehicles}
                      aria-label={isPlaying ? "일시정지" : "재생"}
                      title={isDetectingVehicles ? "차량 탐지 중에는 재생할 수 없습니다"
                        : isPlaying ? "일시정지합니다" : "재생합니다"}
                    >
                      <PlayerIcon name={isPlaying ? "pause" : "play"} size={20} />
                    </button>

                    <div className="volume-control">
                      <button
                        className="control-btn"
                        onClick={toggleMute}
                        aria-label={isMuted || volume === 0 ? "음소거 해제" : "음소거"}
                        title={isMuted || volume === 0 ? "음소거를 해제합니다" : "소리를 끕니다"}
                      >
                        <PlayerIcon
                          name={isMuted || volume === 0 ? "volumeMute" : volume < 50 ? "volumeLow" : "volumeHigh"}
                        />
                      </button>
                      <input
                        className="volume-slider"
                        type="range"
                        min="0"
                        max="100"
                        value={isMuted ? 0 : volume}
                        style={{ "--fill": `${isMuted ? 0 : volume}%` }}
                        onChange={(e) => {
                          const next = Number(e.target.value);
                          setVolume(next);
                          setIsMuted(next === 0);
                        }}
                        aria-label="음량"
                      />
                    </div>

                    <span className="time-display">
                      {formatTime(Math.floor(currentTime))}
                      <span className="time-sep">/</span>
                      {formatTime(Math.floor(playerDuration))}
                    </span>

                    <div className="right-controls">
                      <span className="cam-now-chip" ref={camChipRef} hidden />
                      <select
                        className="speed-select"
                        value={playbackSpeed}
                        onChange={(e) => setPlaybackSpeed(e.target.value)}
                        aria-label="재생 속도"
                        title="재생 속도를 바꿉니다"
                      >
                        <option value="0.25">0.25x</option>
                        <option value="0.5">0.5x</option>
                        <option value="1">1x</option>
                        <option value="1.5">1.5x</option>
                        <option value="2">2x</option>
                      </select>
                      {selectedVideo.events.some((e) => e.hasClip && e.clipStartSec != null) && (
                        <div className="clip-view-toggle" role="group" aria-label="사고 구간 화면">
                          <button
                            type="button"
                            className={clipView === "orig" ? "active" : ""}
                            aria-pressed={clipView === "orig"}
                            onClick={() => setClipView("orig")}
                            title="빨간 구간에서도 원본 영상을 봅니다"
                          >원본</button>
                          <button
                            type="button"
                            className={clipView === "cam" ? "active" : ""}
                            aria-pressed={clipView === "cam"}
                            onClick={() => setClipView("cam")}
                            title="사고 구간에서 AI가 충돌로 본 부위를 표시합니다"
                          >충돌 표시</button>
                        </div>
                      )}
                      {/* 전체화면에선 영화관 모드가 의미 없어 숨긴다 */}
                      {!isFullscreen && (
                        <button
                          className={`control-btn ${isTheaterMode ? "active" : ""}`}
                          onClick={() => setIsTheaterMode((enabled) => !enabled)}
                          disabled={isChoosingVehicle}
                          aria-label="영화관 모드"
                          title={isChoosingVehicle ? "차량을 고르는 중에는 쓸 수 없습니다"
                            : isTheaterMode ? "영화관 모드를 끕니다" : "영화관 모드로 크게 봅니다"}
                        >
                          <PlayerIcon name="theater" />
                        </button>
                      )}
                      <button
                        className="control-btn"
                        onClick={handleToggleFullscreen}
                        disabled={isChoosingVehicle && !isFullscreen}
                        aria-label={isFullscreen ? "전체화면 종료" : "전체화면"}
                        title={isChoosingVehicle && !isFullscreen ? "차량을 고르는 중에는 쓸 수 없습니다"
                          : isFullscreen ? "전체화면을 종료합니다" : "전체화면으로 봅니다"}
                      >
                        <PlayerIcon name={isFullscreen ? "exitFullscreen" : "fullscreen"} />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            {/* 오른쪽 패널: (1280px 이상) 진행 단계 + 이 영상의 분석 이력. 넓은 화면에서 닫으면(헤더의 '패널 닫기') 칸 폭이
                0.3초 동안 0 으로 줄며 사라진다(App.css) — 그동안 보이도록 닫혀도 화면에서 빼지 않고, 대신 inert 로
                보이지 않는 버튼에 Tab·클릭이 가지 않게 한다. 좁은 화면에선 영상 아래에 늘 보인다(영상 크기와 상관없음) */}
            <div className="watch-right" inert={isXLScreen && !isHistoryOpen} aria-hidden={isXLScreen && !isHistoryOpen ? true : undefined}>
              {isXLScreen && workflowSteps}
              <AnalysisHistoryPanel
                history={selectedVideo.analysisHistory || []}
                events={selectedVideo.events}
                progressText={(taskId) => {
                  const job = analyzingJobs.find((j) => j.taskId === taskId);
                  return job ? `분석 중 · ${jobProgress(job).shortText}` : null;
                }}
                activeNo={highlightAnalysisNo}
                onSelect={(history) => {
                  // 그 분석의 클립을 왼쪽 목록에서 강조(나머지는 흐리게). 같은 이력을 다시 누르면 해제
                  setHighlightAnalysisNo((cur) => (cur === history.no ? null : history.no));
                }}
              />
            </div>
          </div>
        </main>
      )}

      {!selectedVideo && <TeamFooter />}
      {/* 설정 모달 */}
      {isSettingsOpen && (
        <div className="settings-modal-overlay" onClick={() => setIsSettingsOpen(false)}>
          <div className="settings-modal-container" role="dialog" aria-modal="true" aria-label="계정 설정" onClick={(e) => e.stopPropagation()}>
            <div className="settings-modal-sidebar">
              <span className="account-settings-kicker" translate="no">PERSONAL SPACE</span><h3>계정 설정</h3><div className="settings-identity"><span>{avatarInitial}</span><strong>{displayName}</strong><small>{roleLabel}</small></div>
              <button
                className={`settings-tab-btn ${activeSettingsTab === "account" ? "active" : ""}`}
                onClick={() => setActiveSettingsTab("account")}
              >
                <InterfaceIcon name="user" /> 내 정보 설정
              </button>
              <button
                className={`settings-tab-btn ${activeSettingsTab === "security" ? "active" : ""}`}
                onClick={() => setActiveSettingsTab("security")}
              >
                <InterfaceIcon name="lock" /> 비밀번호 변경
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
                  <span className="account-settings-kicker" translate="no">MY ACCOUNT</span>
                  <h2>{currentUser?.role === "ADMIN" ? "관리자 정보 설정" : "내 정보 설정"}</h2>
                  <p className="tab-description">
                    {currentUser?.role === "ADMIN"
                      ? "관리자 기본 정보를 확인 및 수정할 수 있습니다."
                      : "회원 기본 정보를 확인 및 수정할 수 있습니다."}
                  </p>

                  <div className="settings-form-group">
                    <label htmlFor="account-username">계정 아이디</label>
                    <input id="account-username" type="text" value={currentUsername} disabled className="disabled-input" />
                  </div>

                  <div className="settings-form-group">
                    <label htmlFor="account-name">이름</label>
                    <input id="account-name"
                      type="text"
                      value={profileName}
                      onChange={(e) => setProfileName(e.target.value)}
                      placeholder="이름 입력"
                    />
                  </div>

                  <div className="settings-form-group">
                    <label htmlFor="account-phone">연락처</label>
                    <input id="account-phone"
                      type="text"
                      value={profilePhone}
                      onChange={(e) => setProfilePhone(e.target.value)}
                      placeholder="연락처 입력 (예: 010-1234-5678)"
                    />
                  </div>

                  <div className="settings-form-group">
                    <label htmlFor="account-email">이메일 주소</label>
                    <input id="account-email"
                      type="email"
                      value={profileEmail}
                      onChange={(e) => setProfileEmail(e.target.value)}
                      placeholder="이메일 입력"
                    />
                  </div>

                  <button
                    className="settings-save-btn"
                    onClick={async () => {
                      if (!profileName.trim()) {
                        alert("이름을 입력해 주세요.");
                        return;
                      }
                      try {
                        const updated = await api.updateMe({
                          name: profileName.trim(),
                          phone: profilePhone.trim(),
                          email: profileEmail.trim(),
                        });
                        if (onUpdateUser) onUpdateUser(updated);
                        showToast("사용자 정보가 성공적으로 저장되었습니다.", "success");
                        alert("사용자 정보가 성공적으로 저장되었습니다.");
                      } catch (err) {
                        alert(`정보 수정 실패: ${err.message}`);
                      }
                    }}
                  >
                    수정 내용 저장
                  </button>
                </div>
              )}

              {activeSettingsTab === "security" && (
                <div className="settings-tab-content">
                  <span className="account-settings-kicker" translate="no">MY ACCOUNT</span>
                  <h2>비밀번호 변경</h2>
                  <p className="tab-description">시스템 보안을 위해 주기적으로 비밀번호를 변경해 주십시오.</p>

                  <div className="settings-form-group">
                    <label htmlFor="account-current-password">현재 비밀번호</label>
                    <input id="account-current-password"
                      type="password"
                      placeholder="현재 비밀번호 입력"
                      value={currentPassword}
                      onChange={(e) => setCurrentPassword(e.target.value)}
                    />
                  </div>

                  <div className="settings-form-group">
                    <label htmlFor="account-new-password">새 비밀번호</label>
                    <input id="account-new-password"
                      type="password"
                      placeholder="새 비밀번호 입력"
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                    />
                  </div>

                  <div className="settings-form-group">
                    <label htmlFor="account-confirm-password">새 비밀번호 확인</label>
                    <input id="account-confirm-password"
                      type="password"
                      placeholder="새 비밀번호 다시 입력"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                    />
                  </div>

                  <button
                    className="settings-save-btn"
                    onClick={async () => {
                      if (!currentPassword || !newPassword || !confirmPassword) {
                        alert("모든 필드를 입력해 주세요.");
                        return;
                      }
                      if (newPassword !== confirmPassword) {
                        alert("새 비밀번호가 서로 일치하지 않습니다.");
                        return;
                      }
                      if (newPassword.length < 4) {
                        alert("새 비밀번호는 최소 4자 이상이어야 합니다.");
                        return;
                      }
                      try {
                        await api.changePassword({
                          current_password: currentPassword,
                          new_password: newPassword,
                        });
                        showToast("비밀번호가 성공적으로 변경되었습니다.", "success");
                        alert("비밀번호가 성공적으로 변경되었습니다.");
                        setCurrentPassword("");
                        setNewPassword("");
                        setConfirmPassword("");
                        setIsSettingsOpen(false);
                      } catch (err) {
                        alert(`비밀번호 변경 실패: ${err.message}`);
                      }
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

      {/* 영상 정보 팝업 (영상 목록 카드의 ⋮) — 화면 가운데, Esc·바깥 클릭·닫기로 닫힘 */}
      {infoVideo && (() => {
        const v = infoVideo;
        const up = formatAnalyzedAt(v.uploadedAt);
        const lastStatus = v.lastTask ? (HISTORY_STATUS[v.lastTask.status]?.label || v.lastTask.status) : null;
        const rows = [
          ["녹화일자", v.date || "-"],
          ["업로드 일시", up ? `${up[0]} ${up[1]}` : "-"],
          ["원본 파일", v.fileName || "-"],
          ["영상 길이", formatTime(v.duration)],
          ["해상도", v.width && v.height ? `${v.width}×${v.height}${v.fps ? ` · ${Math.round(v.fps)}fps` : ""}` : "-"],
          ["사고 의심 클립", `${v.events.length}건`],
          ["마지막 분석", lastStatus || "아직 없음"],
        ];
        return (
          <div className="settings-modal-overlay" onClick={() => setInfoVideo(null)}>
            <div
              className="video-info-modal"
              role="dialog"
              aria-modal="true"
              aria-labelledby="video-info-title"
              onClick={(e) => e.stopPropagation()}
              onKeyDown={(e) => {
                if (e.key === "Escape") setInfoVideo(null);
              }}
            >
              <div className="video-info-modal-header">
                <h2 id="video-info-title" title={getVideoDisplayTitle(v, videos)}>
                  {getVideoDisplayTitle(v, videos)}
                </h2>
                <button className="clip-modal-close" onClick={() => setInfoVideo(null)} autoFocus aria-label="닫기">
                  ✕
                </button>
              </div>
              <dl className="video-info-list">
                {rows.map(([k, val]) => (
                  <div key={k}>
                    <dt>{k}</dt>
                    <dd title={val}>{val}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
        );
      })()}

      {/* 사고 클립 삭제 확인 (1건 · 전체) — 화면 가운데. 기본 초점은 '취소'(Enter 실수로 지우지 않게), Esc·바깥 클릭으로 닫힘 */}
      {(pendingDeleteEvent || pendingClearAll) && (() => {
        const isAll = pendingClearAll;
        const closeConfirm = () => {
          setPendingDeleteEvent(null);
          setPendingClearAll(false);
        };
        return (
          <div className="settings-modal-overlay" onClick={closeConfirm}>
            <div
              className="confirm-modal"
              role="alertdialog"
              aria-modal="true"
              aria-labelledby="confirm-modal-title"
              aria-describedby="confirm-modal-desc"
              onClick={(e) => e.stopPropagation()}
              onKeyDown={(e) => {
                if (e.key === "Escape") closeConfirm();
              }}
            >
              <div className="confirm-modal-icon" aria-hidden="true">
                <TrashIcon size={24} />
              </div>
              <h2 id="confirm-modal-title" className="confirm-modal-title">
                {isAll
                  ? `사고 의심 클립 ${selectedVideo?.events.length ?? 0}개를 모두 삭제하시겠습니까?`
                  : "클립 영상을 삭제하시겠습니까?"}
              </h2>
              <p id="confirm-modal-desc" className="confirm-modal-desc">
                {isAll
                  ? "이 영상에서 감지된 모든 클립 영상과 감지 기록이 삭제됩니다."
                  : "해당 사고 구간의 클립 영상과 감지 기록이 삭제됩니다."}
                <br />
                삭제한 뒤에는 되돌릴 수 없습니다.
              </p>
              <div className="upload-modal-actions">
                <button
                  className="confirm-danger-btn"
                  onClick={() => {
                    const id = pendingDeleteEvent?.id;
                    closeConfirm();
                    if (isAll) handleClearAllEvents();
                    else handleDeleteEvent(id);
                  }}
                >
                  {isAll ? "전체 삭제" : "삭제"}
                </button>
                <button
                  className="upload-cancel-btn"
                  autoFocus
                  onClick={closeConfirm}
                >
                  취소
                </button>
              </div>
            </div>
          </div>
        );
      })()}

      {/* 사고구간 클립 팝업 (+ 원본 클립 다운로드) */}
      {clipEvent && (
        <div className="settings-modal-overlay clip-viewer-overlay" onClick={() => setClipEvent(null)}>
          <div
            className="clip-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="clip-modal-title"
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => {
              if (e.key === "Escape") setClipEvent(null);
            }}
          >
            <div className="clip-modal-header">
              <div className="clip-modal-heading">
                <h2 id="clip-modal-title" className="clip-modal-title">사고구간 클립</h2>
                <div className="clip-modal-chips">
                  <span className="clip-chip">
                    ⏱ {formatTime(clipEvent.timestamp)}
                    {clipEvent.endTimestamp != null ? ` ~ ${formatTime(clipEvent.endTimestamp)}` : ""}
                  </span>
                  {clipEvent.prob != null && (
                    <span className="clip-chip clip-chip-danger">
                      사고 확률 {(clipEvent.prob * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
              </div>
              <button
                className="clip-close-btn"
                onClick={() => setClipEvent(null)}
                aria-label="클립 창 닫기"
                autoFocus
              >
                ✕
              </button>
            </div>

            <div className="clip-video-frame">
              <video className="clip-modal-video" src={api.clipUrl(clipEvent.id)} controls autoPlay />
            </div>

            <div className="clip-modal-footer">
              <div className="clip-legend">
                <span className="clip-legend-item">
                  <i className="clip-legend-swatch clip-legend-a" /> 빨간 테두리 — 사고가 의심되는 순간이에요
                </span>
                <span className="clip-legend-item">
                  <i className="clip-legend-swatch clip-legend-s" /> 초록 테두리 — 특별한 이상이 없는 장면이에요
                </span>
              </div>
              {clipEvent.hasRawClip ? (
                <a
                  className="clip-download-btn"
                  href={api.rawClipUrl(clipEvent.id)}
                  download
                  title="박스·상태 표시가 없는 원본 사고 구간 영상을 내려받습니다"
                >
                  ⬇ 원본 클립 다운로드
                </a>
              ) : (
                <span className="clip-download-unavailable" title="이 영상을 다시 분석하면 원본 클립이 만들어집니다">
                  원본 클립 없음 · 다시 분석하면 생성
                </span>
              )}
            </div>
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
      {/* 동일 날짜 여러 영상 선택 모달 */}
      {datePickerVideos && (
        <DateVideosModal
          dateText={datePickerVideos.dateText}
          videos={datePickerVideos.videos}
          onClose={() => setDatePickerVideos(null)}
          onSelectVideo={(video) => {
            setDatePickerVideos(null);
            handleWatchVideo(video);
          }}
        />
      )}
    </div>
  );
}

function UploadModal({ onClose, onUploaded, onError }) {
  const [filesList, setFilesList] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({ current: 0, total: 0, currentName: "" });
  const [dragActive, setDragActive] = useState(false);

  const fileInputRef = useRef(null);

  const addFiles = (newFiles) => {
    if (!newFiles || newFiles.length === 0) return;
    const todayStr = formatDate(new Date());
    const items = Array.from(newFiles).map((file, idx) => ({
      id: `${Date.now()}_${idx}_${Math.random().toString(36).substring(2, 7)}`,
      file,
      recordingDate: todayStr,
      title: "", // 선택 — 비우면 '녹화일 녹화본 #N'
    }));
    setFilesList((prev) => [...prev, ...items]);
  };

  const handleFileChange = (e) => {
    addFiles(e.target.files);
    if (e.target) e.target.value = "";
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files);
    }
  };

  const removeFile = (id) => {
    setFilesList((prev) => prev.filter((item) => item.id !== id));
  };

  const updateFileDate = (id, dateStr) => {
    setFilesList((prev) =>
      prev.map((item) => (item.id === id ? { ...item, recordingDate: dateStr } : item))
    );
  };

  const updateFileTitle = (id, title) => {
    setFilesList((prev) => prev.map((item) => (item.id === id ? { ...item, title } : item)));
  };

  const updateAllDates = (dateStr) => {
    setFilesList((prev) => prev.map((item) => ({ ...item, recordingDate: dateStr })));
  };

  const handleUpload = async () => {
    if (filesList.length === 0) {
      onError("업로드할 영상 파일을 선택해 주세요.");
      return;
    }
    setUploading(true);
    let successCount = 0;
    let failCount = 0;

    for (let i = 0; i < filesList.length; i += 1) {
      const item = filesList[i];
      setUploadProgress({
        current: i + 1,
        total: filesList.length,
        currentName: item.file.name,
      });
      try {
        await api.uploadVideo(item.file, item.recordingDate || null, item.title);
        successCount += 1;
      } catch (e) {
        failCount += 1;
        console.error(`Failed to upload ${item.file.name}:`, e);
      }
    }

    setUploading(false);
    if (successCount > 0) {
      if (failCount > 0) {
        onError(`${successCount}개 업로드 완료 (${failCount}개 실패)`);
      }
      onUploaded();
    } else {
      onError("모든 영상 파일 업로드에 실패했습니다.");
    }
  };

  return (
    <div className="settings-modal-overlay" onClick={onClose}>
      <div className="upload-modal multi-upload-modal" onClick={(e) => e.stopPropagation()}>
        <div className="upload-modal-header">
          <div>
            <h2>영상 다중 업로드</h2>
            <p className="tab-description">분석할 CCTV 녹화 영상을 업로드합니다. 여러 파일을 한번에 선택하고 파일마다 날짜와 제목을 지정할 수 있습니다.</p>
          </div>
          <button className="clip-modal-close" onClick={onClose} disabled={uploading}>✕</button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          multiple
          style={{ display: "none" }}
          onChange={handleFileChange}
        />

        {filesList.length === 0 ? (
          <div
            className={`upload-dropzone ${dragActive ? "drag-active" : ""}`}
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="dropzone-icon">📹</div>
            <p className="dropzone-title">클릭하여 영상 파일 선택 (복수 선택 가능)</p>
            <p className="dropzone-sub">또는 여기에 영상 파일들을 드래그 앤 드롭하세요</p>
          </div>
        ) : (
          <div className="upload-file-list-container">
            <div className="upload-file-list-toolbar">
              <span className="file-count-badge">선택된 영상: <strong>{filesList.length}개</strong></span>
              {/* 파일 추가는 목록 아래 끌어다 놓기 칸(클릭해도 선택)으로 한다 */}
              <div className="file-toolbar-actions">
                <button
                  type="button"
                  className="clear-all-files-btn"
                  onClick={() => setFilesList([])}
                  disabled={uploading}
                >
                  전체 삭제
                </button>
              </div>
            </div>

            <div className="bulk-date-bar">
              <span>일괄 녹화일자 지정:</span>
              <input
                type="date"
                max={formatDate(new Date())}
                onChange={(e) => {
                  if (e.target.value) updateAllDates(e.target.value);
                }}
                disabled={uploading}
              />
              <span className="bulk-date-tip">(각 파일별 개별 일자 설정 가능)</span>
            </div>

            <div className="upload-file-scroll-list">
              {filesList.map((item) => (
                <div key={item.id} className="upload-file-item">
                  <div className="file-item-info">
                    <span className="file-item-icon">🎥</span>
                    <div className="file-item-name-group">
                      <span className="file-item-name" title={item.file.name}>{item.file.name}</span>
                      <span className="file-item-size">{(item.file.size / (1024 * 1024)).toFixed(1)} MB</span>
                    </div>
                  </div>

                  <div className="file-item-date-group">
                    <label>녹화일자:</label>
                    <input
                      type="date"
                      value={item.recordingDate}
                      max={formatDate(new Date())}
                      onChange={(e) => updateFileDate(item.id, e.target.value)}
                      disabled={uploading}
                    />
                  </div>

                  <button
                    type="button"
                    className="file-item-remove-btn"
                    onClick={() => removeFile(item.id)}
                    disabled={uploading}
                    title="이 파일을 업로드 목록에서 뺍니다"
                  >
                    ✕
                  </button>

                  {/* 제목(선택) — 파일 이름·녹화일자 줄 아래 한 줄 */}
                  <div className="file-item-title-group">
                    <label htmlFor={`upload-title-${item.id}`}>제목:</label>
                    <input
                      id={`upload-title-${item.id}`}
                      type="text"
                      value={item.title}
                      maxLength={100}
                      placeholder={`입력하지 않으면 '${item.recordingDate || "녹화일"} 녹화본'으로 표시됩니다`}
                      onChange={(e) => updateFileTitle(item.id, e.target.value)}
                      disabled={uploading}
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* 파일이 있어도 끌어다 놓아 더 추가할 수 있는 작은 칸 (클릭하면 파일 선택) */}
            <div
              className={`upload-dropzone upload-dropzone-compact ${dragActive ? "drag-active" : ""} ${uploading ? "disabled" : ""}`}
              onDragEnter={uploading ? undefined : handleDrag}
              onDragOver={uploading ? undefined : handleDrag}
              onDragLeave={uploading ? undefined : handleDrag}
              onDrop={uploading ? undefined : handleDrop}
              onClick={() => {
                if (!uploading) fileInputRef.current?.click();
              }}
            >
              <span className="dropzone-compact-icon" aria-hidden="true">＋</span>
              <span>여기에 영상 파일을 끌어다 놓거나 클릭해서 추가</span>
            </div>
          </div>
        )}

        {uploading && (
          <div className="upload-progress-box">
            <div className="upload-progress-info">
              <span>업로드 진행 중 ({uploadProgress.current}/{uploadProgress.total})</span>
              <span className="upload-progress-filename">{uploadProgress.currentName}</span>
            </div>
            <div className="analysis-progress-track">
              <div
                className="analysis-progress-fill"
                style={{ width: `${(uploadProgress.current / uploadProgress.total) * 100}%` }}
              />
            </div>
          </div>
        )}

        <div className="upload-modal-actions">
          <button
            className="upload-submit-btn"
            onClick={handleUpload}
            disabled={uploading || filesList.length === 0}
          >
            {uploading
              ? `업로드 중 (${uploadProgress.current}/${uploadProgress.total})...`
              : `${filesList.length > 0 ? `${filesList.length}개 ` : ""}영상 업로드`}
          </button>
          <button className="upload-cancel-btn" onClick={onClose} disabled={uploading}>
            취소
          </button>
        </div>
      </div>
    </div>
  );
}

function DateVideosModal({ dateText, videos, onClose, onSelectVideo }) {
  if (!videos || videos.length === 0) return null;

  return (
    <div className="settings-modal-overlay" onClick={onClose}>
      <div className="date-videos-modal" onClick={(e) => e.stopPropagation()}>
        <div className="date-videos-header">
          <div>
            <h2>📅 {dateText} 영상 목록 ({videos.length}개)</h2>
            <p className="tab-description">
              해당 날짜에 총 {videos.length}개의 영상이 존재합니다. 시청할 영상을 선택해 주세요.
            </p>
          </div>
          <button className="clip-modal-close" onClick={onClose}>✕</button>
        </div>

        <div className="date-videos-scroll-list">
          {videos.map((v) => (
            <div key={v.id} className="date-video-card-item" onClick={() => onSelectVideo(v)}>
              <div className="date-video-thumb-wrapper">
                <img
                  className="video-thumbnail-img"
                  src={api.thumbnailUrl(v.id)}
                  alt={`${v.date} 썸네일`}
                  onError={(e) => { e.currentTarget.style.display = "none"; }}
                />
                <span className="event-count-badge">의심 구간 {v.events.length}건</span>
              </div>
              <div className="date-video-card-info">
                <h3>{getVideoDisplayTitle(v, videos)}</h3>
                <div className="date-video-card-meta">
                  <span>⏱ 영상 길이: {formatTime(v.duration)}</span>
                </div>
              </div>
              <button className="date-video-play-btn">
                ▶ 시청하기
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function RequireAuth({ children }) {
  return getToken() ? children : <Navigate to="/login" replace />;
}

export default function App() {
  const [currentUser, setCurrentUser] = useState(() => getStoredUser());
  const [user, setUser] = useState(() => !!getToken());
  // 로그인돼 있으면 화면을 그리기 전에 세션부터 연다 — 영상·썸네일·클립은 이때 받는
  // 쿠키로 열리므로(헤더를 못 싣는 <video>·<img>), 먼저 그리면 첫 썸네일이 막힌다
  const [sessionReady, setSessionReady] = useState(() => !getToken());

  useEffect(() => {
    if (!getToken()) return;
    api.startSession().then((u) => {
      setCurrentUser(u);
      saveAuth(getToken(), u);
    }).catch(() => { })   // 만료(401)면 api.js 가 로그인 화면으로 보낸다
      .finally(() => setSessionReady(true));
  }, []);

  const handleUpdateUser = (updatedUser) => {
    setCurrentUser(updatedUser);
    saveAuth(getToken(), updatedUser);
  };

  const handleLogout = () => {
    api.logout().catch(() => { });   // 영상 쿠키도 지운다
    clearAuth();
    setCurrentUser(null);
    setUser(false);
  };

  if (!sessionReady) return null;

  return (

    <BrowserRouter>
      <Routes>
        <Route
          path="/login"
          element={
            <LoginPage
              onLogin={(loggedUser) => {
                setCurrentUser(loggedUser);
                setUser(true);
              }}
            />
          }
        />
        <Route path="/signup" element={<SignupPage />} />
        <Route
          path="/videos"
          element={
            <RequireAuth>
              <Dashboard
                currentUser={currentUser}
                onUpdateUser={handleUpdateUser}
                onLogout={handleLogout}
                view="list"
              />
            </RequireAuth>
          }
        />
        <Route
          path="/videos/:videoId"
          element={
            <RequireAuth>
              <Dashboard
                currentUser={currentUser}
                onUpdateUser={handleUpdateUser}
                onLogout={handleLogout}
                view="watch"
              />
            </RequireAuth>
          }
        />
        <Route path="*" element={<Navigate to={user ? "/videos" : "/login"} replace />} />
      </Routes>
    </BrowserRouter>
  );
}
