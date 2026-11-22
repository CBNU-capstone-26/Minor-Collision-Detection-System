// FastAPI 백엔드 연동 클라이언트.
// vite 프록시(/api → http://localhost:8000)를 통해 호출한다.

const TOKEN_KEY = "psd_token";
const USER_KEY = "psd_user";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}
export function getStoredUser() {
  const raw = localStorage.getItem(USER_KEY);
  return raw ? JSON.parse(raw) : null;
}
export function saveAuth(token, user) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}
export function clearAuth() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export async function retryRequest(operation, { attempts = 3, delayMs = 500, shouldRetry = () => true } = {}) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      if (!shouldRetry(error)) throw error;
      if (attempt < attempts) {
        await new Promise((resolve) => setTimeout(resolve, delayMs * attempt));
      }
    }
  }
  throw lastError;
}

export function isRetryableRequestError(error) {
  return error.status == null || error.status === 429 || error.status >= 500;
}

export async function withRequestTimeout(operation, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await operation(controller.signal);
  } finally {
    clearTimeout(timeout);
  }
}

async function request(path, { method = "GET", body, isForm = false, signal } = {}) {
  const headers = {};
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  let payload = body;
  if (body && !isForm) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const res = await fetch(`/api${path}`, { method, headers, body: payload, signal });
  // 로그인이 만료됐거나 토큰이 유효하지 않으면 저장된 로그인 정보를 지우고 로그인 화면으로.
  // (로그인·회원가입 자체의 401 — 아이디/비밀번호 오류 — 는 화면에 그대로 보여 준다)
  if (res.status === 401 && !path.startsWith("/auth/login") && !path.startsWith("/auth/signup")) {
    clearAuth();
    if (!window.location.pathname.startsWith("/login")) {
      window.location.replace("/login?expired=1");
    }
    const error = new Error("로그인이 만료되었습니다. 다시 로그인해 주세요.");
    error.status = 401;
    throw error;
  }
  if (!res.ok) {
    let detail = `요청 실패 (${res.status})`;
    try {
      const data = await res.json();
      detail = data.detail || detail;
    } catch {
      /* ignore */
    }
    const error = new Error(detail);
    error.status = res.status;
    throw error;
  }
  if (res.status === 204) return null;
  return res.json();
}

// ---------- 데이터 정규화 (API → UI 형태) ----------
function normalizeEvent(e) {
  return {
    id: e.id,
    timestamp: Math.round(e.timestamp_sec || 0),
    endTimestamp:
      e.end_timestamp_sec != null ? Math.round(e.end_timestamp_sec) : null,
    // 반올림 전 값 — 재생바의 사고 구간 위치·이동처럼 1초보다 정확해야 하는 곳에 쓴다
    startSec: e.timestamp_sec || 0,
    endSec: e.end_timestamp_sec ?? null,
    analyzedAt: e.analyzed_at || null, // 분석이 끝나 클립이 만들어진 시각(ISO, UTC)
    analysisNo: e.analysis_no ?? null, // 이 영상의 몇 번째 분석에서 나온 클립인지 (한 번의 분석 = 같은 번호)
    clipStartSec: e.clip_start_sec ?? null, // 클립이 원본 영상에서 시작하는 시각
    prob: e.crash_prob,
    hasClip: !!e.has_clip,
    hasRawClip: !!e.has_raw_clip, // 합성 없는 원본 사고 클립을 내려받을 수 있는지
    title: "사고 의심 구간",
    status:
      e.crash_prob != null && e.crash_prob >= 0.5 ? "확인 필요" : "분석 완료",
  };
}

export function normalizeVideo(v) {
  const created = v.created_at || "";
  return {
    id: v.id,
    title: v.title || "", // 업로드할 때 넣은 제목 (없으면 화면이 '녹화일 녹화본 #N')
    videoType: v.video_type || null,
    collisionLabel: v.collision_label || null,
    fileName: v.video_name || "", // 업로드한 원본 파일 이름
    uploadedAt: v.created_at || null, // 업로드 시각(ISO, UTC)
    // 보관 기간이 끝나 영상·클립이 지워질 때까지 남은 날(서버 시계 기준). 보관 기간을 끄면 null
    expiresInDays: v.expires_in_days ?? null,
    date: v.recording_date || (created ? created.slice(0, 10) : ""),
    duration: Math.round(v.duration_sec || 0),
    width: v.width,
    height: v.height,
    fps: v.fps,
    events: (v.events || []).map(normalizeEvent),
    // 분석을 한 번이라도 완료했는지 — 다시 분석할 때 확인 문구를 바꾼다
    analyzedBefore: !!v.analysis_done,
    // 가장 최근 분석 작업 — 실패·중단이면 화면이 '다시 실행'을 안내한다
    lastTask: v.last_task
      ? { id: v.last_task.task_id, status: v.last_task.status, errorMessage: v.last_task.error_message }
      : null,
    // 분석 이력(요청 순) — 영상 상세에서만 온다. 시각은 ISO(UTC)
    analysisHistory: (v.analysis_history || []).map((h) => ({
      no: h.analysis_no,
      taskId: h.task_id,
      status: h.status,
      errorMessage: h.error_message,
      requestedAt: h.requested_at,
      finishedAt: h.finished_at,
      eventCount: h.event_count, // 찾은 클립 수 (이력 기능 전 분석은 null)
    })),
  };
}

// ---------- 인증 ----------
export const api = {
  signup: (data) => request("/auth/signup", { method: "POST", body: data }),
  login: (data) => request("/auth/login", { method: "POST", body: data }),
  // 화면을 새로 열 때 — 최신 사용자 정보 + 영상·썸네일·클립용 쿠키(로그인 때도 받는다)
  startSession: () => request("/auth/session", { method: "POST" }),
  logout: () => request("/auth/logout", { method: "POST" }),
  updateMe: (data) => request("/auth/me", { method: "PUT", body: data }),
  changePassword: (data) => request("/auth/change-password", { method: "POST", body: data }),

  // ---------- 영상 ----------
  listVideos: async (days) => {
    const q = days ? `?days=${days}` : "";
    const list = await retryRequest(() =>
      withRequestTimeout((signal) => request(`/videos${q}`, { signal }), 7000));
    return list.map(normalizeVideo);
  },
  getVideo: async (id) => normalizeVideo(await request(`/videos/${id}`)),
  uploadVideo: (file, recordingDate, title, videoType, collisionLabel) => {
    const form = new FormData();
    form.append("file", file);
    if (recordingDate) form.append("recording_date", recordingDate);
    if (title && title.trim()) form.append("title", title.trim());
    if (videoType) form.append("video_type", videoType);
    if (collisionLabel) form.append("collision_label", collisionLabel);
    return request("/videos", { method: "POST", body: form, isForm: true });
  },
  updateVideoType: async (id, videoType) => normalizeVideo(await request(`/videos/${id}/type`, { method: "PATCH", body: { video_type: videoType } })),
  updateVideoTitle: async (id, title) => normalizeVideo(await request(`/videos/${id}/title`, { method: "PATCH", body: { title } })),
  updateCollisionLabel: async (id, collisionLabel) => normalizeVideo(await request(`/videos/${id}/collision-label`, { method: "PATCH", body: { collision_label: collisionLabel } })),
  streamUrl: (id) => `/api/videos/${id}/stream`,
  thumbnailUrl: (id) => `/api/videos/${id}/thumbnail`,
  deleteVideo: (id) => request(`/videos/${id}`, { method: "DELETE" }),
  deleteVideos: async (ids) => {
    const results = await Promise.allSettled(ids.map((id) => request(`/videos/${id}`, { method: "DELETE" })));
    const failedIds = ids.filter((_, index) => results[index].status === "rejected");
    return { deletedCount: ids.length - failedIds.length, failedIds };
  },
  clearAnalysisHistory: (id) => request(`/videos/${id}/analysis-history`, { method: "DELETE" }),
  deleteEvent: (videoId, eventId) =>
    request(`/videos/${videoId}/events/${eventId}`, { method: "DELETE" }),
  clearEvents: (videoId) =>
    request(`/videos/${videoId}/events`, { method: "DELETE" }),

  // ---------- 분석 ----------
  detectVehicles: (videoId, timestampSec = 0, signal) =>
    request(`/videos/${videoId}/detect-vehicles?timestamp_sec=${timestampSec}`, {
      method: "POST",
      signal,
    }),
  getVehicleDetectionTask: (videoId, taskId, signal) =>
    request(`/videos/${videoId}/vehicle-detection-tasks/${taskId}`, { signal }),
  analyze: (videoId, bbox, signal) =>
    request(`/videos/${videoId}/analyze`, { method: "POST", body: bbox, signal }),
  getTaskStatus: (taskId) => retryRequest(
    () => withRequestTimeout((signal) => request(`/tasks/${taskId}`, { signal }), 7000),
    { shouldRetry: isRetryableRequestError },
  ),
  cancelTask: (taskId) => request(`/tasks/${taskId}/cancel`, { method: "POST" }),
  // 로그인한 사용자의 대기·진행 중 분석(창을 다시 열 때 진행 상황 복원용)
  listActiveTasks: () => request("/tasks/active"),
  // 실패·취소된 분석을 같은 영상·같은 차량 박스로 다시 실행
  retryTask: (taskId) => request(`/tasks/${taskId}/retry`, { method: "POST" }),
  clipUrl: (eventId) => `/api/events/${eventId}/clip`,
  // 박스·상태 띠 없이 사고 구간만 자른 원본 클립(다운로드)
  rawClipUrl: (eventId) => `/api/events/${eventId}/clip/raw`,
};
