import unittest
from unittest.mock import MagicMock
from fastapi import HTTPException
from app.api_schemas import SignupRequest
from app.routers.auth import signup


class SignupValidationTests(unittest.TestCase):
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
