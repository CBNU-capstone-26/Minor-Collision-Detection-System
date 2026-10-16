import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import './EyeComfort.css';

const STORAGE_KEY = 'siot.eye-comfort.v1';
function readSettings() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY));
    return {
      enabled: value?.enabled === true,
      strength: typeof value?.strength === 'number' && Number.isFinite(value.strength)
        ? Math.max(0, Math.min(100, value.strength)) : 40,
    };
  } catch {
    return { enabled: false, strength: 40 };
  }
}

export default function EyeComfort() {
  const [settings, setSettings] = useState(readSettings);
  const [open, setOpen] = useState(false);
  const [target, setTarget] = useState(() => document.fullscreenElement || document.body);
  const trigger = useRef(null);

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(settings)); }
    catch { /* Private browsing/storage restrictions must not block the control. */ }
  }, [settings]);

  useEffect(() => {
    const change = () => setTarget(document.fullscreenElement || document.body);
    document.addEventListener('fullscreenchange', change);
    return () => document.removeEventListener('fullscreenchange', change);
  }, []);

  useEffect(() => {
    if (!open) return;
    const escape = (event) => {
      if (event.key === 'Escape') {
        setOpen(false);
        trigger.current?.focus();
      }
    };
    document.addEventListener('keydown', escape);
    return () => document.removeEventListener('keydown', escape);
  }, [open]);

  return createPortal(<>
    <div className="eye-comfort-tint" aria-hidden="true"
      style={{ opacity: settings.enabled ? settings.strength / 100 * 0.65 : 0 }} />
    <aside className="eye-comfort" aria-label="화면 눈 보호 설정">
      {open && <section id="eye-comfort-panel" className="eye-comfort-panel" aria-label="눈 보호 모드">
        <div className="eye-comfort-heading">
          <strong>눈 보호 모드</strong>
          <button type="button" onClick={() => { setOpen(false); trigger.current?.focus(); }} aria-label="눈 보호 설정 닫기">×</button>
        </div>
        <p>이 웹 화면을 따뜻한 색으로 조절합니다.</p>
        <label className="eye-comfort-toggle">
          <span>따뜻한 화면</span>
          <input type="checkbox" role="switch" checked={settings.enabled}
            onChange={e => setSettings(s => ({ ...s, enabled: e.target.checked }))} />
        </label>
        <label className="eye-comfort-strength" htmlFor="eye-comfort-strength">
          <span>필터 강도</span><output>{settings.strength}%</output>
        </label>
        <input id="eye-comfort-strength" type="range" min="0" max="100" step="5"
          value={settings.strength} disabled={!settings.enabled}
          onChange={e => setSettings(s => ({ ...s, strength: Number(e.target.value) }))} />
        <p className="eye-comfort-note">영상·CAM 색상을 정확히 비교할 때는 꺼주세요. 원본 영상과 분석 결과는 바뀌지 않습니다.</p>
      </section>}
      <button ref={trigger} type="button" className="eye-comfort-trigger"
        aria-expanded={open} aria-controls="eye-comfort-panel" onClick={() => setOpen(v => !v)}>
        <span aria-hidden="true">◐</span> 눈 보호 <span className="eye-comfort-status">{settings.enabled ? '켜짐' : '꺼짐'}</span>
      </button>
    </aside>
  </>, target);
}
