class database:
    def __init__(self, userId):
        self.startTime = time.time()
        self.bucketCounter = 1
        self.readBucketCounter=4;
        self.deleteBucketCounter = 2;
        self.minuteCounter = 1
        self.db=self.init_db()
        self.userId = userId
        self.x_total = 0;
        self.y_total=0;
        self.N = 0
        self.varXList=[]
        self.varYList=[]

    def init_db(self):
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

    def push(self,x,y):
        currentTime=time.time()
        diff = int(currentTime - self.startTime)
        #for now test ke liye hr 10 second ke mean push kar ra
        diff = diff//10
        if(diff < self.minuteCounter):
            self.x_total += x
            self.y_total += y
            self.N +=1
        else:
            #push new data
            x_mean = (self.x_total/self.N)
            y_mean = (self.y_total/self.N)
            data = {"name": self.userId,"x_coord":x_mean, "y_coord":y_mean}
            bucket = str(self.bucketCounter) + "min"
            self.db.child(bucket).push(data)
            #calculate variance from last bucket
            readBucket = str(self.readBucketCounter) + "min"
            varX,varY = self.calculate(readBucket)
            self.varXList.append(varX)
            self.varYList.append(varY)
            #delete old data
            delBucket = str(self.deleteBucketCounter) + "min"
            self.db.child(delBucket).remove()

            self.update()


    def update(self):
        self.x_total=0
        self.y_total=0
        self.N=0
        self.minuteCounter += 1
        self.bucketCounter = self.bucketCounter % 4
        self.bucketCounter += 1
        self.readBucketCounter = self.readBucketCounter % 4
        self.readBucketCounter += 1
        self.deleteBucketCounter = self.deleteBucketCounter % 4
        self.deleteBucketCounter += 1


    def calculate(self,bucket):
        print(bucket)
        mean_x,dev_x,mean_y,dev_y,N=0,0,0,0,0
        user_val=[]
        all_users = self.db.child(bucket).get()
        try:
            for user in all_users.each():
                N+=1
                val=user.val()
                mean_x+=(val['x'])
                mean_y+=(val['y'])
                if(val['name']==userId):
                    user_val.append(val['x'])
                    user_val.append(val['y'])
        except:
            N=1
        mean_x/=N
        mean_y/=N

        if(len(user_val)!=0):
            dev_x+=((user_val[0]-mean_x)**2)/N
            dev_y+=((user_val[1]-mean_y)**2)/N
        return dev_x,dev_y

    def delete(self,bucket):
        db.child(bucket).remove()