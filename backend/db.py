import json
import os

# Simulating SQLAlchemy without the library
class MockQuery:
    def __init__(self, cls):
        self.cls = cls
        self._filters = {}

    def _get_data(self):
        return self.cls._load_all()

    def filter_by(self, **kwargs):
        new_q = MockQuery(self.cls)
        new_q._filters = {**self._filters, **kwargs}
        return new_q

    def _execute(self):
        data = self._get_data()
        filtered = []
        for item in data:
            match = True
            for k, v in self._filters.items():
                if getattr(item, k) != v:
                    match = False
                    break
            if match:
                filtered.append(item)
        return filtered

    def first(self):
        res = self._execute()
        return res[0] if res else None

    def all(self):
        return self._execute()

    def count(self):
        return len(self._execute())

class MockUserMeta(type):
    @property
    def query(cls):
        return MockQuery(cls)

class User(metaclass=MockUserMeta):
    def __init__(self, name="", email="", password="", phone="", id=None):
        self.id = id
        self.name = name
        self.email = email
        self.password = password
        self.phone = phone

    @staticmethod
    def _get_db_path():
        # Use a path relative to this file
        return os.path.join(os.path.dirname(__file__), "users_mock.json")

    @classmethod
    def _load_all(cls):
        path = cls._get_db_path()
        if not os.path.exists(path):
            # Try to load from users.json if it exists (the original JSON)
            orig_path = os.path.join(os.path.dirname(__file__), "users.json")
            if os.path.exists(orig_path):
                try:
                    with open(orig_path, "r") as f:
                        data = json.load(f)
                        return [cls(**u) for u in data]
                except:
                    pass
            return []
        try:
            with open(path, "r") as f:
                data = json.load(f)
                return [cls(**u) for u in data]
        except:
            return []

    def __repr__(self):
        return f"<User {self.email}>"

class MockSession:
    def __init__(self):
        self.to_add = []

    def add(self, obj):
        self.to_add.append(obj)

    def commit(self):
        users = User._load_all()
        for u in self.to_add:
            if not u.id:
                u.id = len(users) + 1
            # Check if already exists by email
            exists = False
            for existing in users:
                if existing.email == u.email:
                    existing.name = u.name
                    existing.password = u.password
                    existing.phone = u.phone
                    exists = True
                    break
            if not exists:
                users.append(u)
        
        path = User._get_db_path()
        data = [u.__dict__ for u in users]
        with open(path, "w") as f:
            json.dump(data, f)
        self.to_add = []

class db:
    Model = object
    session = MockSession()
    
    @staticmethod
    def init_app(app):
        pass

    @staticmethod
    def create_all():
        path = User._get_db_path()
        if not os.path.exists(path):
            # Populate from original users.json if possible
            users = User._load_all()
            data = [u.__dict__ for u in users]
            with open(path, "w") as f:
                json.dump(data, f)
