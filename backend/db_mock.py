import json
import os

# Simulating SQLAlchemy without the library
class MockQuery:
    def __init__(self, cls):
        self.cls = cls
        self.data = self.cls._load_all()

    def filter_by(self, **kwargs):
        filtered = []
        for item in self.data:
            match = True
            for k, v in kwargs.items():
                if getattr(item, k) != v:
                    match = False
                    break
            if match:
                filtered.append(item)
        self.data = filtered
        return self

    def first(self):
        return self.data[0] if self.data else None

    def all(self):
        return self.data

    def count(self):
        return len(self.data)

class User:
    query = None # Will be set below
    
    def __init__(self, name="", email="", password="", phone="", id=None):
        self.id = id
        self.name = name
        self.email = email
        self.password = password
        self.phone = phone

    @staticmethod
    def _get_db_path():
        return os.path.join(os.path.dirname(__file__), "users_mock.json")

    @classmethod
    def _load_all(cls):
        path = cls._get_db_path()
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r") as f:
                data = json.load(f)
                return [cls(**u) for u in data]
        except:
            return []

    @classmethod
    def _save_all(cls, users):
        path = cls._get_db_path()
        data = [u.__dict__ for u in users]
        with open(path, "r") as f:
             # Need to handle ID assignment if we were serious, but this is a mock
             pass
        # simpler save
        with open(path, "w") as f:
            json.dump(data, f)

    def __repr__(self):
        return f"<User {self.email}>"

# Set query property
User.query = MockQuery(User) # This is a bit recursive in the mock but works for .first()

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
            users.append(u)
        # Very crude save
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
            with open(path, "w") as f:
                json.dump([], f)
