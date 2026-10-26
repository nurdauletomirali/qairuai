"""8-практикалық жұмыс. Іске қосу: python Practice8_solution.py
Қажетті пакеттер: python -m pip install numpy scikit-learn
"""
from fractions import Fraction as F
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

for rain in (1, 0):
    sh = F(1,2)*F(5,6)*(F(4,6) if rain else F(2,6))
    sl = F(1,2)*F(2,6)*(F(1,6) if rain else F(5,6))
    ph = sh/(sh+sl)
    loss = {'Extra': (1-ph)*60, 'Usual': ph*30}
    utility = {k: -v for k,v in loss.items()}
    print('Rain =', rain, 'P(High) =', ph, float(ph))
    print('EL:', {k:float(v) for k,v in loss.items()})
    print('EU:', {k:float(v) for k,v in utility.items()}, 'Action:', max(utility,key=utility.get))
assert F(10,11)+F(1,11)==1
print('Laplace:', F(1,8), F(7,8))
X = np.array([100,150,200,250,300]).reshape(-1,1)
y = np.array([29,41,53,66,78])
linear = LinearRegression().fit(X,y)
print('w, b:', linear.coef_, linear.intercept_)
print('180:', linear.predict([[180]]))
yp = linear.predict(X[:2])
print('First two:', yp, 'residuals:', y[:2]-yp, 'MSE:', np.mean((y[:2]-yp)**2))
for name,w,b in [('A',.20,10),('B',.25,5)]:
    pred=w*X[:2,0]+b
    print(name,pred,np.mean((y[:2]-pred)**2))
X_train=np.array([[1,1],[1,1],[1,1],[1,0],[1,0],[0,1],[1,1],[1,0],[0,0],[0,0],[0,0],[0,0]])
y_train=np.array([1,1,1,1,1,1,0,0,0,0,0,0])
model=make_pipeline(StandardScaler(),LogisticRegression())
model.fit(X_train,y_train)
p=model.predict_proba(np.array([[1,1],[1,0]]))[:,1]
print('Logistic probabilities:',p)
print('alerts:',(p>=.50).astype(int))
print('Boundary distances:',p-.50)
TP,FP,FN,TN=21,9,3,27
accuracy=F(TP+TN,TP+FP+FN+TN);precision=F(TP,TP+FP);recall=F(TP,TP+FN)
f1=2*precision*recall/(precision+recall)
for name,value in [('Accuracy',accuracy),('Precision',precision),('Recall',recall),('F1',f1)]:
    print(name, f'{float(value)*100:.1f}%')
assert accuracy==F(4,5) and f1==F(7,9)
print('All Low: TP=0 FP=0 FN=5 TN=45; Accuracy=90.0%; Recall=0.0%')
