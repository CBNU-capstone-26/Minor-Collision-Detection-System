import unittest

from app.backup_runner import validate_repository


class RepositoryValidationTest(unittest.TestCase):
    def test_rejects_source_and_insecure_destination(self):
        for repository in ('s3:https://r2.example/source/backup',
                           's3:http://r2.example/backup/repo'):
            with self.subTest(repository=repository), self.assertRaises(ValueError):
                validate_repository(repository, 'https://r2.example', 'source')

    def test_accepts_separate_bucket(self):
        validate_repository('s3:https://r2.example/capstone26-backup/repo',
                            'https://r2.example', 'source')
