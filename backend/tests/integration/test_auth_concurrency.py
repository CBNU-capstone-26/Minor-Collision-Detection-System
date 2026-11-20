"""Opt-in concurrent signup tests against a disposable local database."""
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from app import api_schemas, db_models
from app.db_connection import Base
from app.routers import auth


@unittest.skipUnless(os.environ.get('SIOT_TEST_AUTH_DATABASE_URL'), 'isolated database required')
class SignupConcurrencyTests(unittest.TestCase):
    def setUp(self):
        url = os.environ['SIOT_TEST_AUTH_DATABASE_URL']
        parsed = urlparse(url)
        if parsed.hostname not in ('127.0.0.1', 'localhost', '::1') or not parsed.path.lstrip('/').startswith('siot_test_'):
            raise RuntimeError('requires a disposable local siot_test_ database')
        self.engine = create_engine(url)
        self.addCleanup(self.engine.dispose)
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine, expire_on_commit=False)

    def run_simultaneous(self, requests, column):
        barrier = threading.Barrier(2)
        lock, count = threading.Lock(), [0]

        def synchronize(conn, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith('SELECT') and f'WHERE users.{column} ' in statement:
                with lock:
                    count[0] += 1
                    number = count[0]
                if number <= 2:
                    barrier.wait(timeout=5)

        def signup(req):
            with self.sessions() as db:
                try:
                    auth.signup(api_schemas.SignupRequest(**req), db)
                    return 200
                except HTTPException as exc:
                    return exc.status_code

        event.listen(self.engine, 'after_cursor_execute', synchronize)
        try:
            with patch.object(auth, 'hash_password', return_value='test-hash'), ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(signup, requests))
        finally:
            event.remove(self.engine, 'after_cursor_execute', synchronize)
        self.assertEqual(sorted(results), [200, 400])

    def test_duplicate_usernames_are_reported_as_client_error(self):
        self.run_simultaneous([
            {'username':'same','name':'A','password':'test'},
            {'username':'same','name':'B','password':'test'},
        ], 'username')

    def test_duplicate_emails_are_reported_as_client_error(self):
        self.run_simultaneous([
            {'username':'first','name':'A','email':'same@example.test','password':'test'},
            {'username':'second','name':'B','email':'same@example.test','password':'test'},
        ], 'email')

    def test_duplicate_profile_email_updates_are_reported_as_client_error(self):
        with self.sessions.begin() as db:
            first = db_models.User(username='first', password_hash='test')
            second = db_models.User(username='second', password_hash='test')
            db.add_all([first, second])
            db.flush()
            ids = [first.id, second.id]
        barrier, lock, count = threading.Barrier(2), threading.Lock(), [0]

        def synchronize(conn, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith('SELECT') and 'WHERE users.email ' in statement:
                with lock:
                    count[0] += 1
                    number = count[0]
                if number <= 2:
                    barrier.wait(timeout=5)

        def update(user_id):
            with self.sessions() as db:
                user = db.get(db_models.User, user_id)
                try:
                    auth.update_me(api_schemas.UpdateUserRequest(email='same@example.test'), user, db)
                    return 200
                except HTTPException as exc:
                    return exc.status_code

        event.listen(self.engine, 'after_cursor_execute', synchronize)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(update, ids))
        finally:
            event.remove(self.engine, 'after_cursor_execute', synchronize)
        self.assertEqual(sorted(results), [200, 400])
