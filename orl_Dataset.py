import numpy as np
from sklearn import metrics,svm
import matplotlib.pyplot as plt
import matplotlib.image as mimg

training_data=np.zeros((40*8,92))
train_target = np.zeros((40*8))
test_data=np.zeros((2*40,92))
test_target = np.zeros((40*2))

c1=-1;c2=-1
#FOR TRAINING DATA
for i in range(1,41):
    for j in range(1,11):
        add='./orl_faces/'+'s'+str(i)+'/'+str(j)+'.pgm'
        if(j<=8):
            c1=c1+1
            im=mimg.imread(add)
            val=im.mean(axis=0)
            val=val.reshape(1,-1) # feature of a single image 
            training_data[c1,:]=val
            train_target[c1]=i
        else:
            c2=c2+1
            im=mimg.imread(add)
            val=im.mean(axis=0)
            val=val.reshape(1,-1)
            test_data[c2,:]=val
            test_target[c2]=i
            
print(train_target)            
svm_model=svm.SVC(kernel='rbf')
svm_model=svm_model.fit(training_data,train_target)
op=svm_model.predict(test_data)
print(metrics.accuracy_score(test_target,op))  

      
