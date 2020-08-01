
def init_db():
    import pyrebase
    config = {
        "apiKey": "bccdb0a053ff6210f4944c55f87fc6d241e199b0",
        "authDomain": "data-5eef0.firebaseapp.com",
        "databaseURL": "https://data-5eef0.firebaseio.com/",
        "storageBucket": "data-5eef0.appspot.com",
    }

    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    return db


class DataBase:
    def __init__(self, teacher_name: str) -> None:
        self.teacher_name = teacher_name
        self.db = init_db()
        self.counterDict = {}
        self.avgList = {}
        self.db.child("nameList").child("teacher_name").push(self.teacher_name)
        self.db.child("online").push(True)

    def insert_data(self, name, score):
        old_avg = 0
        n = 1
        try:
            old_avg = self.avgList[name]
            n = self.counterDict[name]
        except:
            old_avg = score
            self.counterDict[name] = n
        new_avg = old_avg + ((score - old_avg) / n)
        self.avgList[name] = new_avg
        data = {name: new_avg}
        self.db.child(self.teacher_name).update(data)
        self.counterDict[name] += 1

    def end_ses(self):
        self.db.child("online").push(False)
