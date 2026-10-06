"""Download public Drive files without accepting arbitrary remote URLs."""
import re
import time
from email.message import Message
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit

import requests
from fastapi import HTTPException

MAX_BYTES = 10 * 1024 ** 3


def drive_download_url(link):
    parsed = urlsplit(link.strip())
    if parsed.scheme != 'https' or parsed.hostname != 'drive.google.com' or parsed.port not in (None, 443) or parsed.username:
        raise ValueError('https://drive.google.com/ 형식의 영상 공유 링크를 입력해 주세요.')
    query = parse_qs(parsed.query)
    match = re.fullmatch(r'/file/d/([\w-]+)(?:/view|/preview)?/?', parsed.path)
    file_id = match.group(1) if match else query.get('id', [''])[0] if parsed.path in ('/open', '/uc') else ''
    if not re.fullmatch(r'[A-Za-z0-9_-]{10,200}', file_id):
        raise ValueError('폴더가 아닌 개별 영상 파일의 공유 링크를 입력해 주세요.')
    params = {'id': file_id, 'export': 'download'}
    if query.get('resourcekey'):
        params['resourcekey'] = query['resourcekey'][0]
    return 'https://drive.usercontent.google.com/download?' + urlencode(params)


def allowed_download(url):
    p = urlsplit(url)
    return (p.scheme == 'https' and p.port in (None, 443) and not p.username
            and (p.hostname in ('drive.google.com', 'drive.usercontent.google.com')
                 or (p.hostname or '').endswith('.googleusercontent.com')))


class ConfirmationForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}
        self.active = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'form' and attrs.get('id') == 'download-form':
            self.active = True
            self.action = attrs.get('action')
        if tag == 'input' and self.active and attrs.get('name'):
            self.fields[attrs['name']] = attrs.get('value', '')

    def handle_endtag(self, tag):
        if tag == 'form':
            self.active = False


def download_drive_video(link, destination):
    try:
        url = drive_download_url(link)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    started = time.monotonic()
    try:
        with requests.Session() as session:
            for _ in range(8):
                if not allowed_download(url):
                    raise HTTPException(400, '공개 다운로드 링크가 아닙니다. Drive 공유 권한을 확인해 주세요.')
                with session.get(url, stream=True, allow_redirects=False, timeout=(15, 60)) as response:
                    if response.is_redirect:
                        url = urljoin(url, response.headers['Location'])
                        continue
                    if response.status_code != 200:
                        raise HTTPException(400, 'Drive 파일을 가져올 수 없습니다. 공유 권한이나 다운로드 제한을 확인해 주세요.')
                    if 'text/html' in response.headers.get('Content-Type', '').lower():
                        page = bytearray()
                        for chunk in response.iter_content(65536):
                            page.extend(chunk)
                            if len(page) > 1024 * 1024:
                                break
                        form = ConfirmationForm()
                        form.feed(page.decode('utf-8', errors='replace'))
                        if not form.action:
                            raise HTTPException(400, '공유 설정을 ‘링크가 있는 모든 사용자’로 지정하고 다운로드를 허용해 주세요.')
                        url = urljoin(url, form.action) + '?' + urlencode(form.fields)
                        continue
                    if int(response.headers.get('Content-Length', '0')) > MAX_BYTES:
                        raise HTTPException(413, '최대 10GB 영상까지 가져올 수 있습니다.')
                    header = Message()
                    header['Content-Disposition'] = response.headers.get('Content-Disposition', '')
                    filename = (header.get_filename() or 'drive-video.mp4').replace('\\', '/').split('/')[-1][:255]
                    size = 0
                    with destination.open('wb') as output:
                        for chunk in response.iter_content(1024 * 1024):
                            size += len(chunk)
                            if size > MAX_BYTES:
                                raise HTTPException(413, '최대 10GB 영상까지 가져올 수 있습니다.')
                            if time.monotonic() - started > 3600:
                                raise HTTPException(504, '가져오기 시간이 초과되었습니다. 다시 시도해 주세요.')
                            output.write(chunk)
                    return filename
            raise HTTPException(400, 'Drive 다운로드를 완료하지 못했습니다. 공유 권한과 다운로드 제한을 확인해 주세요.')
    except requests.RequestException as exc:
        raise HTTPException(502, 'Drive 연결에 실패했습니다. 잠시 후 다시 시도해 주세요.') from exc
