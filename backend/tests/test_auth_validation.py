import unittest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from app.api_schemas import SignupRequest, UpdateUserRequest
from app.routers.auth import signup, update_me


class SignupValidationTests(unittest.TestCase):
    def test_empty_signup_email_is_saved_as_null(self):
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        req = SignupRequest(username='valid', name='valid', email='  ', password='test')
        with patch('app.routers.auth.hash_password', return_value='hash'):
            user = signup(req, db)
        self.assertIsNone(user.email)
    def test_profile_can_clear_optional_email(self):
        from types import SimpleNamespace
        db = MagicMock()
        user = SimpleNamespace(id=1, name='test', email='old@example.test', phone=None)
        result = update_me(UpdateUserRequest(email=''), user, db)
        self.assertIsNone(result.email)
        db.commit.assert_called_once()

    def test_omitted_email_leaves_existing_email_unchanged(self):
        from types import SimpleNamespace
        db = MagicMock()
        user = SimpleNamespace(id=1, name='test', email='old@example.test', phone=None)
        result = update_me(UpdateUserRequest(name='new'), user, db)
        self.assertEqual(result.email, 'old@example.test')
    def test_oversized_profile_values_are_rejected_before_database(self):
        from types import SimpleNamespace
        for field, value in [('name', 'x' * 101), ('email', 'x' * 256), ('phone', 'x' * 51)]:
            with self.subTest(field=field):
                db = MagicMock()
                user = SimpleNamespace(id=1, name='test', email=None, phone=None)
                with self.assertRaises(HTTPException) as error:
                    update_me(UpdateUserRequest(**{field: value}), user, db)
                self.assertEqual(error.exception.status_code, 400)
                db.query.assert_not_called()
                db.commit.assert_not_called()
    def test_blank_or_oversized_signup_values_are_rejected_before_database(self):
        base = {'username': 'valid', 'name': 'valid', 'email': None, 'password': 'test'}
        for field, value in [('username', ''), ('username', ' '), ('username', 'x' * 256),
                             ('name', ' '), ('name', 'x' * 101), ('email', 'x' * 256),
                             ('password', '')]:
            with self.subTest(field=field, length=len(value)):
                data = {**base, field: value}
                db = MagicMock()
                db.query.return_value.filter.return_value.first.return_value = None
                with self.assertRaises(HTTPException) as error:
                    signup(SignupRequest(**data), db)
                self.assertEqual(error.exception.status_code, 400)
                db.query.assert_not_called()
