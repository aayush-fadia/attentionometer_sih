from pyasn1.compat.octets import null
from pyasn1.type.univ import Null
import Levenshtein
from datetime import datetime
from statistics import mean


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
    def __init__(self, nameList) -> None:
        self.db = init_db()
        self.db.remove()
        self.scoreDict = {}
        self.oldCategory = {}
        self.oldScore = {}
        self.avg = 0
        self.avgCounter = 0
        self.id = str(datetime.now().strftime("%d-%m-%Y-%H:%M:%S"))
        self.db.child("online").set(self.id)
        for name in nameList:
            self.oldScore[name] = -1
            self.oldCategory[name] = -1

    def insert_data(self, classesDict, attentionDict):
        for name in classesDict.keys():
            if self.oldCategory[name] != classesDict[name]:
                student_data = {name: classesDict[name].value}
                self.db.child("Teacher").update(student_data)
                self.oldCategory[name] = classesDict[name]
            if self.oldScore[name] != attentionDict[name]:
                self.avg += attentionDict[name]
                self.avgCounter += 1
        if self.avgCounter > 5:
            self.db.child("means_score").set(self.avg/self.avgCounter)
            self.avgCounter = 0
            self.avg = 0

    def end_ses(self):
        self.db.child("online").set("null")


"""import random
import time
db = DataBase()
stlist = ["Surbhi", "Sauyma", "Aayush", "Praneeth", "Kanishka", "surbha", "sauwma", "ayush"]
for i in range(20000):
    for s in stlist:
        score = random.randrange(50, 100)
        category = random.randrange(0, 4)
        if random.random() < 0.4:
            score -= 50
        db.insert_data(s, category, score)
    time.sleep(5)"""
#db.end_ses()

"""
    def checkName(self, name: str):
        parts = name.split()
        for n in self.NameList:
            n_split = n.split()
            if Levenshtein.distance(n, name) < 5 or Levenshtein.distance(parts[0], n_split[0]) < 2:
                return n
        self.NameList.append(name)
        return name
"""