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
    def __init__(self) -> None:
        # self.teacher_name = teacher_name
        self.db = init_db()
        self.db.remove()
        self.scoreDict = {}
        
        self.NameList = []
        self.id = str(datetime.now().strftime("%d-%m-%Y-%H:%M:%S"))
        self.db.child("online").set(self.id)

    def checkName(self, name: str):
        parts = name.split()
        for n in self.NameList:
            n_split = n.split()
            if Levenshtein.distance(n, name) < 5 or Levenshtein.distance(parts[0], n_split[0]) < 2:
                return n
        self.NameList.append(name)
        return name

    def insert_data(self, name, category, score):
        name = self.checkName(name)
        student_data = {name : category}
        self.db.child("Teacher").update(student_data)
        try:
            self.scoreDict[name].append(score)
        except KeyError:
            self.scoreDict[name] = [score]

        push = True
        avg = 0
        for ques in self.scoreDict.values():
            if len(ques) == 0:
                push = False
                break
            else:
                avg += ques.pop(0)
        if push:
            self.db.child("means_score").set(avg/len(self.scoreDict))



    def end_ses(self):
        self.db.child("online").set("null")


"""import random
import time
db = DataBase()
stlist = ["Surbhi", "Sauyma", "Aayush", "Praneeth", "Kanishka", "surbha", "sauwma", "ayush"]
for i in range(200):
    for s in stlist:
        score = random.randrange(50, 100)
        category = random.randrange(0, 3)
        if random.random() < 0.4:
            score -= 50
        db.insert_data(s, category, score)
    time.sleep(5)
#db.end_ses()
"""
